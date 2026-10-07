#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import time
import pytest
import numpy as np
import spark

pytest.importorskip('PySide6', reason='the run viewer needs PySide6')
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from run_viewer import _brain, SIGNAL

# The tests of this module share fixtures computed once per worker: with pytest-xdist and --dist loadgroup,
# they run on one worker.
pytestmark = pytest.mark.xdist_group('workspace')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

R = spark.recording
EPISODES = {0: ('X', (2, 3, 1)), 1: ('X', (1, 1, 4)), 2: ('Y', (3, 3, 3)), 3: (None, (2, 2, 2))}
"""
    Experiment of every run, by seed, and the calls of 5 steps of each of its three episodes.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='module')
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp('project')
    brain = _brain()
    rate = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
    for seed, (experiment, lengths) in EPISODES.items():
        recorder = R.Recorder(root, brain, [R.Measurements('summary', rate, group='episode', trigger=R.Always())], experiment=experiment,
                              hparams={'seed': seed, 'lr': 0.5 if experiment == 'Y' else 0.1})
        runner = R.Runner(brain, recorder)
        for episode, calls in enumerate(lengths):
            recorder.tag(episode=episode)
            for _ in range(calls):
                # Logged at the step the call starts at.
                recorder.log(loss=float(seed + episode))
                runner.run(5, {'signal': SIGNAL})
            recorder.log({'episode/steps': 5.0 * calls}, tag='episode')
        runner.close()
    return root

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def project(qapp, root):
    from spark.graph_editor.runs.workspace import Project
    return Project(root)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _by_seed(project):
    return {run.hparams['seed']: run for run in project.runs}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestProject:

    def test_the_runs_and_their_fields(self, project) -> None:
        runs = _by_seed(project)
        assert sorted(runs) == [0, 1, 2, 3]
        assert project.fields() == ['experiment', 'name', 'lr', 'seed']
        assert [project.value(runs[seed], 'experiment') for seed in range(4)] == ['X', 'X', 'Y', None]
        assert project.value(runs[2], 'lr') == 0.5 and project.value(runs[0], 'name') == 'Brain'

    def test_an_experiment_set_in_the_viewer_leaves_the_run_as_recorded(self, project) -> None:
        runs = _by_seed(project)
        project.set_experiment(runs[3], 'Y')
        project.set_experiment(runs[0], None)
        assert project.experiment(runs[3]) == 'Y' and project.experiment(runs[0]) is None
        assert R.Run(runs[3].path).experiment is None and R.Run(runs[0].path).experiment == 'X'

    def test_runs_added_are_found(self, qapp, root, tmp_path) -> None:
        import shutil
        from spark.graph_editor.runs.workspace import Project
        project = Project(tmp_path)
        assert not project.runs and not project.refresh()
        shutil.copytree(next(root.iterdir()), tmp_path / 'copied')
        changed = []
        project.changed.connect(lambda: changed.append(1))
        assert project.refresh() and len(project.runs) == 1 and changed

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestSpaces:

    def test_the_natural_space_of_a_series(self, project) -> None:
        from spark.graph_editor.runs.workspace import SeriesStore
        store = SeriesStore()
        spaces = store.spaces(_by_seed(project)[0])
        assert spaces.tags() == ['episode']
        assert spaces.natural('episode/steps') == 'episode' and spaces.natural('loss') == 'step'
        assert spaces.natural('summary/first_pool.soma:spikes/active_fraction') == 'episode'

    def test_steps_and_tag_values_map_both_ways(self, project) -> None:
        from spark.graph_editor.runs.workspace import SeriesStore
        spaces = SeriesStore().spaces(_by_seed(project)[0])
        # Episodes of 2, 3 and 1 calls of 5 steps: from steps 0, 10 and 25, to 30.
        np.testing.assert_array_equal(spaces.to_space(np.array([0, 9, 10, 24, 25, 29]), 'episode'), [0, 0, 1, 1, 2, 2])
        assert [spaces.to_steps(episode, 'episode') for episode in (0, 1, 2)] == [(0, 10), (10, 25), (25, 30)]
        assert spaces.to_steps(7, 'episode') is None and spaces.to_steps(12.4, 'step') == (12, 13)
        wall = spaces.to_space(np.array([0, 10, 29]), 'wall')
        assert wall[0] == 0 and np.all(np.diff(wall) >= 0)

    def test_a_series_in_its_natural_space_and_in_others(self, project) -> None:
        from spark.graph_editor.runs.workspace import SeriesStore
        store, run = SeriesStore(), _by_seed(project)[0]
        x, y = store.series(run, 'episode/steps')
        np.testing.assert_array_equal(x, [0, 1, 2])
        np.testing.assert_array_equal(y, [10.0, 15.0, 5.0])
        # Logged per step, averaged per episode in the space of the tag.
        x, y = store.series(run, 'loss', 'episode')
        np.testing.assert_array_equal(x, [0, 1, 2])
        np.testing.assert_array_equal(y, [0.0, 1.0, 2.0])
        x, _ = store.series(run, 'loss')
        np.testing.assert_array_equal(x, [0, 5, 10, 15, 20, 25])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestBands:

    def test_runs_line_up_on_the_values_of_a_tag(self) -> None:
        from spark.graph_editor.runs.workspace import aggregate
        band = aggregate([(np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])), (np.array([1, 2, 3]), np.array([4.0, 6.0, 8.0]))], None)
        np.testing.assert_array_equal(band.x, [0, 1, 2, 3])
        np.testing.assert_array_equal(band.mean, [1.0, 3.0, 4.5, 8.0])
        np.testing.assert_array_equal(band.low, [1.0, 2.0, 3.0, 8.0])
        np.testing.assert_array_equal(band.high, [1.0, 4.0, 6.0, 8.0])
        np.testing.assert_array_equal(band.count, [1, 2, 2, 1])

    def test_runs_are_averaged_in_buckets_of_steps(self) -> None:
        from spark.graph_editor.runs.workspace import aggregate
        first = (np.arange(100.0), np.zeros(100))
        second = (np.arange(0.0, 100.0, 2.0), np.ones(50))
        band = aggregate([first, second], buckets=10)
        assert len(band.x) == 10 and np.all(band.count == 2)
        np.testing.assert_allclose(band.mean, 0.5)
        np.testing.assert_allclose(band.high - band.low, 1.0)
        assert not len(aggregate([]).x) and not len(aggregate([(np.zeros(0), np.zeros(0))]).x)

    def test_smoothing(self) -> None:
        from spark.graph_editor.runs.workspace import smooth
        values = np.array([1.0, np.nan, 1.0, 1.0])
        np.testing.assert_array_equal(smooth(values, 0.0), values)
        np.testing.assert_allclose(smooth(values, 0.9)[[0, 2, 3]], 1.0)
        assert np.isnan(smooth(values, 0.9)[1])
        noisy = np.tile([0.0, 2.0], 50)
        assert np.ptp(smooth(noisy, 0.9)[20:]) < np.ptp(noisy)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestSelection:

    def test_runs_of_an_experiment_are_one_line(self, project) -> None:
        from spark.graph_editor.runs.workspace import Selection
        selection = Selection(project)
        runs = _by_seed(project)
        assert selection.visible == {str(run.path) for run in project.runs} and selection.group_by == ('experiment',)
        lines = {line.label: line for line in selection.lines()}
        assert set(lines) == {'X · 2 runs', 'Y · 1 run', selection.label(runs[3])}
        assert lines['X · 2 runs'].group and not lines[selection.label(runs[3])].group
        assert len({line.color.name() for line in lines.values()}) == 3
        selection.set_visible([runs[1].path], False)
        assert 'X · 1 run' in {line.label for line in selection.lines()}
        selection.set_group_by([])
        assert len(selection.lines()) == 3 and not any(line.group for line in selection.lines())
        selection.set_group_by(['lr'])
        assert {line.label for line in selection.lines()} == {'0.1 · 2 runs', '0.5 · 1 run'}

    def test_a_group_is_its_mean_between_its_lowest_and_highest_run(self, project) -> None:
        from spark.graph_editor.runs.workspace import Selection, SeriesStore
        selection, store = Selection(project), SeriesStore()
        group = next(line for line in selection.lines() if line.label == 'X · 2 runs')
        space, band, series = selection.draw(store, group, 'episode/steps')
        # Episodes of 2, 3, 1 and of 1, 1, 4 calls of 5 steps.
        assert space == 'episode' and len(series) == 2
        np.testing.assert_array_equal(band.x, [0, 1, 2])
        np.testing.assert_array_equal(band.mean, [7.5, 10.0, 12.5])
        np.testing.assert_array_equal(band.low, [5.0, 5.0, 5.0])
        np.testing.assert_array_equal(band.high, [10.0, 15.0, 20.0])
        # In steps, the runs are averaged in buckets.
        selection.set_settings(space='step')
        space, band, _ = selection.draw(store, group, 'episode/steps')
        assert space == 'step' and np.all(band.count >= 1)
        # A run alone is its series.
        alone = next(line for line in selection.lines() if not line.group)
        _, band, series = selection.draw(store, alone, 'loss')
        np.testing.assert_array_equal(band.mean, band.low)
        assert len(series) == 1

    def test_the_settings_and_colors(self, project) -> None:
        from spark.graph_editor.runs.workspace import Selection
        selection = Selection(project)
        changed = []
        selection.changed.connect(lambda: changed.append(1))
        selection.set_settings(smoothing=2.0, members=True)
        assert selection.smoothing == 0.99 and selection.members and len(changed) == 1
        selection.set_settings(smoothing=0.99)
        assert len(changed) == 1
        line = selection.lines()[0]
        selection.set_color(f'group:{line.label.split(" · ")[0]}', QColor('#123456'))
        assert selection.lines()[0].color.name() == '#123456'
        selection.show_newest(2)
        assert len(selection.visible) == 2

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRunsTable:

    @staticmethod
    def _table(project):
        from spark.graph_editor.runs.workspace import Selection
        from spark.graph_editor.runs.workspace_view import RunsTable
        selection = Selection(project)
        return RunsTable(project, selection), selection

    @staticmethod
    def _top(table):
        return {table.tree.topLevelItem(i).text(0): table.tree.topLevelItem(i) for i in range(table.tree.topLevelItemCount())}

    def test_runs_are_listed_by_group_with_the_parameters_that_differ(self, project) -> None:
        table, selection = self._table(project)
        top = self._top(table)
        assert {'X  (2)', 'Y  (1)'} <= set(top) and len(top) == 3
        assert top['X  (2)'].childCount() == 2 and top['X  (2)'].checkState(0) == Qt.CheckState.Checked
        assert table.columns == ['lr', 'seed']
        assert [table.tree.headerItem().text(i) for i in range(6)] == ['Run', 'Created', 'Experiment', 'Steps', 'lr', 'seed']
        selection.set_group_by([])
        assert table.tree.topLevelItemCount() == 4

    def test_an_eye_shows_or_hides_a_run_or_a_group(self, project, qapp) -> None:
        table, selection = self._table(project)
        runs = _by_seed(project)
        self._top(table)['X  (2)'].setCheckState(0, Qt.CheckState.Unchecked)
        qapp.processEvents()
        assert selection.visible == {str(runs[2].path), str(runs[3].path)}
        group = self._top(table)['X  (2)']
        assert group.checkState(0) == Qt.CheckState.Unchecked
        group.child(0).setCheckState(0, Qt.CheckState.Checked)
        qapp.processEvents()
        assert len(selection.visible) == 3 and self._top(table)['X  (2)'].checkState(0) == Qt.CheckState.PartiallyChecked
        assert 'X · 1 run' in {line.label for line in selection.lines()}

    def test_the_search_keeps_the_runs_matching(self, project) -> None:
        table, _ = self._table(project)
        table.search.setText('lr=0.5')
        top = self._top(table)
        assert not top['Y  (1)'].isHidden() and top['X  (2)'].isHidden()

    def test_an_experiment_set_in_the_table(self, project) -> None:
        table, selection = self._table(project)
        runs = _by_seed(project)
        table.set_experiment([str(runs[3].path)], 'Y')
        assert 'Y  (2)' in self._top(table) and 'Y · 2 runs' in {line.label for line in selection.lines()}

    def test_columns_are_resized_by_hand_and_keep_their_widths(self, project, qapp) -> None:
        from PySide6.QtWidgets import QHeaderView
        table, selection = self._table(project)
        header = table.tree.header()
        assert all(header.sectionResizeMode(i) == QHeaderView.ResizeMode.Interactive for i in range(header.count()))
        assert header.stretchLastSection()
        # Until resized by hand, the first column takes the width the others leave, as the table is resized.
        table.resize(560, 400)
        table.show()
        qapp.processEvents()
        from spark.graph_editor.styles.run_viewer import THEME
        assert header.length() <= table.tree.viewport().width() + 1 and table.tree.columnWidth(0) >= THEME.layout.first_column_width
        wide = table.tree.columnWidth(0)
        table.resize(660, 400)
        qapp.processEvents()
        assert table.tree.columnWidth(0) == wide + 100 and not table.state()['widths']
        table.tree.setColumnWidth(2, 173)
        # Built again, as when runs are grouped or columns chosen, the table keeps the width given.
        selection.set_group_by([])
        table.set_columns([*table.columns, 'seed'])
        assert table.tree.columnWidth(2) == 173 and table.state()['widths'] == {'Experiment': 173}
        table.tree.setColumnWidth(0, 150)
        table.resize(760, 400)
        qapp.processEvents()
        assert table.tree.columnWidth(0) == 150
        table.close()

    def test_runs_are_named_by_their_recorder_and_the_time_they_were_created(self, project, tmp_path) -> None:
        import json, shutil
        from spark.graph_editor.runs.workspace import run_names
        table, selection = self._table(project)
        selection.set_group_by([])
        rows = [table.tree.topLevelItem(i) for i in range(table.tree.topLevelItemCount())]
        # The name given to the recorder, here the class of the model, next to the time; not the id of the directory.
        assert {row.text(0) for row in rows} == {'Brain'}
        assert all(re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', row.text(1)) for row in rows)
        # By default, the newest first.
        times = [row.data(1, Qt.ItemDataRole.UserRole) for row in rows]
        assert times == sorted(times, reverse=True)
        # In the legends of the plots, a run is named with its time, the id of its directory left out.
        assert all(project.name(run).startswith('Brain · ') and run.path.name.rsplit('_', 1)[1] not in project.name(run) for run in project.runs)
        # Runs of one name created in the same minute are told apart by the second, then by their order.
        twins = []
        for index, created in enumerate(('2026-10-02T09:15:05-04:00', '2026-10-02T09:15:40-04:00', '2026-10-02T09:15:40-04:00')):
            path = tmp_path / f'20261002-0915{index:02d}_cartpole_{index:08x}'
            shutil.copytree(project.runs[0].path, path)
            R.store.write_json(path / 'run.json', {**json.loads((path / 'run.json').read_text()), 'name': 'cartpole', 'created': created})
            twins.append(R.Run(path))
        assert list(run_names(twins).values()) == ['cartpole · 10-02 09:15:05', 'cartpole · 10-02 09:15:40 (1)', 'cartpole · 10-02 09:15:40 (2)']

    def test_runs_keep_their_places_as_the_table_is_built_again(self, project, qapp) -> None:
        table, selection = self._table(project)
        selection.set_group_by([])
        rows = lambda: [table.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) for i in range(table.tree.topLevelItemCount())]
        created = [str(run.path) for run in project.runs]
        table.tree.sortItems(1, Qt.SortOrder.AscendingOrder)
        assert rows() == created
        # Runs holding the same value, such as the runs of an experiment, follow the order they were created in.
        table.tree.sortItems(2, Qt.SortOrder.AscendingOrder)
        order = rows()
        runs = _by_seed(project)
        experiment_x = [str(runs[0].path), str(runs[1].path)]
        assert [path for path in order if path in experiment_x] == [path for path in created if path in experiment_x]
        # Built again, as when a run is hidden, the table keeps its order.
        table.tree.topLevelItem(0).setCheckState(0, Qt.CheckState.Unchecked)
        qapp.processEvents()
        assert rows() == order

    def test_runs_sort_by_their_steps(self, project) -> None:
        table, selection = self._table(project)
        selection.set_group_by([])
        table.tree.sortItems(3, Qt.SortOrder.AscendingOrder)
        steps = [table.tree.topLevelItem(i).data(3, Qt.ItemDataRole.UserRole) for i in range(4)]
        assert steps == sorted(steps)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestWorkspace:

    @staticmethod
    def _workspace(project):
        from spark.graph_editor.runs.workspace import Selection, SeriesStore
        from spark.graph_editor.runs.workspace_view import WorkspaceView
        selection = Selection(project)
        return WorkspaceView(project, selection, SeriesStore()), selection

    def test_sections_by_the_prefix_of_the_series(self) -> None:
        from spark.graph_editor.runs.workspace_view import section_of
        assert section_of('summary/A_excitatory.soma:spikes/active_fraction') == 'summary · A_excitatory'
        assert section_of('episode/steps') == 'episode' and section_of('loss') == 'logged'
        from spark.graph_editor.runs.workspace_view import in_section
        assert in_section('summary/A_excitatory.soma:spikes/active_fraction') == 'soma:spikes/active_fraction'
        assert in_section('episode/steps') == 'steps' and in_section('loss') == 'loss'

    def test_a_panel_per_series_in_its_space(self, project) -> None:
        workspace, selection = self._workspace(project)
        assert set(workspace.sections) == {'episode', 'logged', 'summary · first_pool'}
        steps = workspace.sections['episode'].panels['episode/steps']
        # Grouped by experiment: X (two runs, with a band), Y (one run) and the run of no experiment.
        assert 'X · 2 runs' in [s[0] for s in steps.plot.series] and len(steps.plot.series) == 3 and len(steps.plot.bands) == 1
        # Named without the prefix of their section, with their space on their horizontal axis.
        assert steps.plot.title == 'steps' and steps.plot.x_label == 'episode'
        assert workspace.sections['logged'].panels['loss'].plot.x_label == 'step'
        selection.set_settings(members=True)
        assert len(steps.plot.faint) == 2
        selection.set_settings(space='step')
        assert steps.plot.x_label == 'step'

    def test_the_settings_bar(self, project) -> None:
        workspace, selection = self._workspace(project)
        assert [workspace._space.itemData(i) for i in range(workspace._space.count())] == ['natural', 'step', 'wall', 'episode']
        workspace._smoothing.setValue(50)
        workspace._members.setChecked(True)
        assert selection.smoothing == 0.5 and selection.members
        workspace.set_columns(3)
        assert workspace.columns == 3

    def test_the_search_and_the_pinned_panels(self, project) -> None:
        workspace, _ = self._workspace(project)
        workspace.set_search('loss')
        assert not workspace.sections['logged'].isHidden() and workspace.sections['episode'].isHidden()
        workspace.set_search('')
        assert not workspace.sections['episode'].isHidden()
        panel = workspace.add_panel(['episode/steps', 'loss'])
        assert not workspace.pinned.isHidden() and any(label.endswith(' · loss') for label, *_ in panel.plot.series)
        workspace.pin(workspace.sections['episode'].panels['episode/steps'])
        assert len(workspace.pinned.panels) == 2
        workspace._unpin(panel)
        assert list(workspace.pinned.panels) == ['episode/steps']

    def test_panels_are_made_when_their_section_opens(self, project) -> None:
        from spark.graph_editor.runs.workspace_view import Section
        workspace, _ = self._workspace(project)
        section = Section('many', [f'key{i}' for i in range(20)], workspace, False)
        assert not section.panels
        section.set_open(True)
        assert len(section.panels) == 20

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestExploration:

    def test_written_and_read_back(self, tmp_path) -> None:
        from spark.graph_editor.runs.workspace import Exploration
        exploration = Exploration(tmp_path / 'kept' / 'view.exploration.json')
        assert exploration.read() is None
        exploration.write({'cursor': 3})
        assert exploration.read() == {'cursor': 3, 'version': Exploration.VERSION}
        assert not list(exploration.path.parent.glob('*.tmp'))
        exploration.path.write_text('{"cursor": 3, "version": 0}')
        assert exploration.read() is None
        exploration.path.write_text('not json')
        assert exploration.read() is None

    def test_refused_within_a_directory_of_runs(self, root, tmp_path) -> None:
        from spark.graph_editor.runs.workspace import Exploration, within_runs
        run = next(path for path in root.iterdir() if (path / 'run.json').exists())
        for path in (root / 'view.json', run / 'view.json', root / 'notes' / 'view.json'):
            assert within_runs(path)
            with pytest.raises(ValueError):
                Exploration(path).write({})
            assert not path.exists()
        assert not within_runs(tmp_path / 'view.json')

    def test_one_per_directory_in_a_folder_of_its_own(self, root, tmp_path, monkeypatch) -> None:
        import spark.graph_editor.runs.workspace as workspace
        monkeypatch.setattr(workspace, 'explorations_path', lambda: tmp_path)
        kept = workspace.Exploration.of(root)
        assert kept.path.parent == tmp_path and kept.path.name.startswith(root.name) and kept.path.name.endswith('.exploration.json')
        (tmp_path / 'b' / root.name).mkdir(parents=True)
        assert workspace.Exploration.of(tmp_path / 'b' / root.name).path != kept.path
        assert workspace.Exploration.of(root).path == kept.path

    def test_the_selection_and_the_experiments_by_name(self, project) -> None:
        from spark.graph_editor.runs.workspace import Project, Selection
        runs = _by_seed(project)
        selection = Selection(project)
        project.set_experiment(runs[3], 'Z')
        selection.set_group_by(['experiment', 'lr'])
        selection.show_only([runs[0].path, runs[3].path])
        selection.set_color(str(runs[3].path), QColor('#123456'))
        selection.set_settings(members=True, smoothing=0.25, space='step')
        state = {'project': project.state(), 'selection': selection.state()}
        assert state['project'] == {'experiments': {runs[3].path.name: 'Z'}}
        assert state['selection']['visible'] == sorted([runs[0].path.name, runs[3].path.name])
        assert state['selection']['colors'][f'run:{runs[3].path.name}'] == '#123456'
        other = Project(project.root)
        restored = Selection(other)
        other.restore(state['project'])
        restored.restore(state['selection'])
        assert other.experiment(other.run(runs[3].path)) == 'Z'
        assert restored.visible == selection.visible and restored.group_by == ('experiment', 'lr')
        assert restored.color(str(runs[3].path)).name() == '#123456'
        assert (restored.members, restored.smoothing, restored.space) == (True, 0.25, 'step')
        assert [line.label for line in restored.lines()] == [line.label for line in selection.lines()]

    def test_the_table_and_the_workspace(self, project) -> None:
        from spark.graph_editor.runs.workspace import Selection, SeriesStore
        from spark.graph_editor.runs.workspace_view import RunsTable, WorkspaceView
        selection = Selection(project)
        table, workspace = RunsTable(project, selection), WorkspaceView(project, selection, SeriesStore())
        table.set_columns(['lr'])
        table.tree.setColumnWidth(0, 210)
        table.search.setText('X')
        workspace.set_columns(3)
        workspace.sections['logged'].set_open(False)
        workspace.add_panel(['loss', 'episode/steps']).set_zoom(None, [0.0, 2.0])
        workspace.sections['episode'].panels['episode/steps'].set_zoom([1.0, 2.0], None)
        state = {'table': table.state(), 'workspace': workspace.state()}
        other_table, other = RunsTable(project, selection), WorkspaceView(project, selection, SeriesStore())
        other_table.restore(state['table'])
        other.restore(state['workspace'])
        assert other_table.columns == ['lr'] and other_table.search.text() == 'X' and other_table.tree.columnWidth(0) == 210
        assert other.columns == 3 and not other.sections['logged'].opened and other.sections['episode'].opened
        assert other.pinned.panels['loss\nepisode/steps'].zoom() == {'x': None, 'y': [0.0, 2.0]}
        assert other.sections['episode'].panels['episode/steps'].zoom() == {'x': [1.0, 2.0], 'y': None}
        other_table.restore({'columns': ['gone', 'seed']})
        assert other_table.columns == ['seed']

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################