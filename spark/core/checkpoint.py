#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import os
import json
import shutil
import hashlib
import tarfile
import pathlib
import tempfile

import jax

import spark.core.utils as utils
from spark.core.config import SparkConfig
from spark.core.backend import split, merge

CHECKPOINT_EXTENSION = '.spark'
"""
    Extension of the files written by `Checkpointable.checkpoint`.
"""

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _local_checkpointer() -> tp.Any:
    """
        Returns an orbax checkpointer, valid only for the current process.
    """
    import orbax.checkpoint as ocp
    index = jax.process_index()
    options = ocp.options.MultiprocessingOptions(primary_host=index, active_processes={index})
    return ocp.Checkpointer(ocp.StandardCheckpointHandler(), multiprocessing_options=options)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _sha256(path: str | os.PathLike) -> str:
    """
        Returns the SHA-256 of a file, as hexadecimal digits.
    """
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _registered_class(cls: type) -> dict[str, str] | None:
    """
        Returns the registry entry of a model or None when the class is not registered.
    """
    from spark.core.registry import REGISTRY, RegistryNamespace
    for namespace in (RegistryNamespace.Components, RegistryNamespace.Interfaces, RegistryNamespace.Neurons):
        entry = getattr(REGISTRY, namespace.name).get_by_cls(cls)
        if entry is not None:
            return {'__module_type__': entry.name, '__subregistry__': namespace.name}
    return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Checkpointable:
    """
        Mixin for creating and loading model checkpoints, used by `SparkModule` and `Controller`.

        The file is a gzipped tar holding model configuration ``model.scfg``, input specifications 
        (in the metadata), and the current model ``state``, written by orbax. 
        
        Model using the mixin need to define a ``config`` and define a ``get_input_specs`` method,
        which provide the input specs the model was built with.
    """

    def checkpoint(self, path: str | os.PathLike, overwrite: bool = False, verbose: bool = True, sha256: bool = False) -> pathlib.Path:
        """
            Saves the model to a ``.spark`` file.

            Parameters
            ----------
            path : str or path-like
                Where to write. ``.spark`` is added to the name when it does not end with it.
            overwrite : bool, default False
                Replace an existing file.
            verbose : bool, default True
                Log where the file was written, and its SHA-256, which `from_checkpoint` can check.
            sha256 : bool, default False
                Export the SHA-256 to a file, ``<file>.sha256`` as ``sha256sum``. 
                We recommend adding a sha256 file when sharing models with other people to prevent
                final users from consuming tampered files.

            Returns
            -------
            pathlib.Path
                The file written.

            Raises
            ------
            RuntimeError
                When the file exists and ``overwrite`` is False, or the model cannot be saved.

            Notes
            -----
            The file is written aside and moved in place once complete. With several processes, each
            process that calls it writes the file on its own.
        """
        import orbax.checkpoint as ocp

        model: tp.Any = self
        tar_path = utils.with_extension(path, CHECKPOINT_EXTENSION)
        try:
            # Check if file exists
            if tar_path.exists() and not overwrite:
                raise RuntimeError(
                    f'Attempting to overwrite file with {tar_path}. Set \"overwrite=True\" if this was intended.'
                )
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            # Written aside, and moved in place once complete.
            temp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f'.{tar_path.stem}.', dir=tar_path.parent))
            partial = temp_dir.with_name(f'{temp_dir.name}.spark')
            try:
                # Save config, with what rebuilding the model needs
                metadata = {'model': _registered_class(type(model)), 'input_specs': model.get_input_specs()}
                model.config.to_file(str(temp_dir / 'model.scfg'), verbose=False, metadata=metadata)
                # Save state
                _, state = split((model))
                checkpointer = _local_checkpointer()
                try:
                    checkpointer.save((temp_dir / 'state').absolute(), args=ocp.args.StandardSave(state))
                finally:
                    checkpointer.close()
                # Compress into tar
                with tarfile.open(partial, 'w:gz') as tar:
                    tar.add(temp_dir, arcname='.')
                os.replace(partial, tar_path)
            finally:
                # Remove temporary files
                shutil.rmtree(temp_dir, ignore_errors=True)
                partial.unlink(missing_ok=True)
            # The SHA-256 beside the file, as sha256sum writes it.
            digest = _sha256(tar_path) if sha256 or verbose else None
            hashed = tar_path.with_name(f'{tar_path.name}.sha256')
            if sha256:
                hashed.write_text(f'{digest}  {tar_path.name}\n')
            else:
                hashed.unlink(missing_ok=True)
        except Exception as e:
            raise RuntimeError(f'Unable to generate checkpoint for model {model.__class__.__name__}: {e}') from e
        # Message
        if verbose:
            print(f'Checkpoint for model {model.__class__.__name__} successfully saved to path: {tar_path} (SHA-256: {digest}).')
        return tar_path

#-----------------------------------------------------------------------------------------------------------------------------------------------#

    # TODO: This is a potentially dangerous operation. We need to add some safe guards to prevent malicious 
    # software to enter a computer unintentionally. 
    # There is a simple SHA-256 to alliviate this issue but what follows relies on the creditibility of the authors ¯\_(ツ)_/¯
    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike, safe: bool = True, verbose: bool = True, sha256: str | None = None) -> tp.Self:
        """
            Loads a model from a ``.spark`` file.

            Parameters
            ----------
            path : str or path-like
                File to read, with or without its ``.spark`` extension.
            safe : bool, default True
                Refuse a file whose configuration or state is not where expected, or that holds links.
            verbose : bool, default True
                Log where the model was read from.
            sha256 : str, optional
                Matching SHA-256 of the file.

            Returns
            -------
            SparkModule or Controller
                A model of the class saved: ``cls`` or a subclass.

            Raises
            ------
            RuntimeError
                When the file cannot be read, does not have the SHA-256 given, or holds a model other
                than a ``cls``.
        """
        import orbax.checkpoint as ocp
        from spark.core.serializer import SparkJSONDecoder

        def is_child(member_name: str, parent: str) -> bool:
            m = pathlib.PurePosixPath(member_name)
            p = pathlib.PurePosixPath(parent)
            # Must be relative
            if m.is_absolute():
                return False
            # Normalize and ensure containment
            try:
                m.relative_to(p)
                return True
            except ValueError:
                return False

        def safe_member(m: tarfile.TarInfo) -> bool:
            return not (m.issym() or m.islnk())

        path = utils.file_with_extension(path, CHECKPOINT_EXTENSION)
        temp_dir = None
        try:
            # A file other than the one published is refused before anything in it is read.
            if sha256 is not None and (found := _sha256(path)) != sha256.strip().lower():
                raise RuntimeError(f'The file has SHA-256 {found}, not {sha256.strip()}: it is not the file published.')
            # Open tar file
            temp_dir = pathlib.Path(tempfile.mkdtemp(prefix='.restore.', dir=pathlib.Path(path).parent))
            with tarfile.open(path, "r:gz") as tar:
                config_file = tar.getmember('./model.scfg')
                state_dir = tar.getmember('./state')
                if safe and not config_file.isfile():
                    raise RuntimeError(
                        f'Unable to validate model.scfg. If you still want to try to extract the file set \"safe=False\"'
                    )
                if safe and not state_dir.isdir():
                    raise RuntimeError(
                        f'Unable to validate state. If you still want to try to extract the file set \"safe=False\"'
                    )
                kwargs = {'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}
                tar.extract(config_file, path=temp_dir, **kwargs)
                state_files = []
                for member in tar.getmembers():
                    if is_child(member.name, state_dir.name):
                        if safe_member(member):
                            state_files.append(member)
                        else:
                            raise RuntimeError(
                                f'A strange possibly malicious file was detected: "{member.name}". If you still want to try to extract the file set "safe=False"'
                            )
                tar.extractall(members=state_files, path=temp_dir, **kwargs)
            # Restore model
            config_path = temp_dir / 'model.scfg'
            config = SparkConfig.from_file(str(config_path.absolute()))
            # The metadata holds encoded classes and specs, read as the configuration is.
            metadata = json.loads(json.dumps(SparkConfig.metadata_from_file(str(config_path))), cls=SparkJSONDecoder)
            # Get model class. Controllers and neurons defined in place are not registered.
            model_cls: type = metadata.get('model') or config.class_ref
            if not issubclass(model_cls, cls):
                raise TypeError(f'The checkpoint holds a {model_cls.__name__}, not a {cls.__name__}.')
            # Initialize the module.
            model = model_cls(config=config)
            dummy_input = {
                port_name: port_spec._create_mock_payload() for port_name, port_spec in metadata['input_specs'].items()
            }
            model(**dummy_input)
            graph, template_state = split((model))
            # Restore state
            checkpointer = _local_checkpointer()
            try:
                state = checkpointer.restore((temp_dir / 'state').absolute(), args=ocp.args.StandardRestore(template_state))
            finally:
                checkpointer.close()
            #Assemble
            model = merge(graph, state)
        except Exception as e:
            raise RuntimeError(f'Unable to restore checkpoint for model {path}: {e}') from e
        finally:
            # Remove temporary files
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        # Message
        if verbose:
            print(f'Model {model.__class__.__name__} loaded successfully from path: {path}.')
        return model

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
