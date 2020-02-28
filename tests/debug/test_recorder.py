import re
from itertools import product

import mltk
import numpy as np

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.debug import *
from tensorkit.debug.recorder import NullRecorder
from tests.helper import *


def standard_recorder_check(ctx, recorder, arrays, expected):
    def assert_eq(a, b):
        if b is None:
            ctx.assertIsNone(a)
        elif isinstance(b, (T.Tensor, np.ndarray, float)):
            assert_allclose(a, b, rtol=1e-4, atol=1e-6)
        else:
            ctx.assertEqual(a, b)

    def f():
        for arr in arrays:
            recorder.record(arr)

        if isinstance(expected, mltk.MetricStats):
            ret = recorder.get()
            for attr in ('mean', 'std'):
                val = getattr(ret, attr)
                expected_val = getattr(expected, attr)
                assert_eq(val, expected_val)
        else:
            assert_eq(recorder.get(), expected)

    ctx.assertIsNone(recorder.get())
    f()

    recorder.clear()
    ctx.assertIsNone(recorder.get())
    f()


class RecorderTestCase(TestCase):

    def test_ConcatRecorder(self):
        arrays = [T.random.randn([2, 4]), T.random.randn([3, 4])]
        standard_recorder_check(
            self,
            ConcatRecorder(),
            arrays,
            T.concat(arrays, axis=0),
        )

    def test_StatsRecorder(self):
        for shape in ([], [4]):
            metrics = mltk.GeneralMetricCollector(shape)
            arrays = [T.random.randn([2, 4]), T.random.randn([3, 4])]
            for arr in arrays:
                metrics.update(T.to_numpy(arr))

            standard_recorder_check(
                self,
                StatsRecorder(shape),
                arrays,
                metrics.stats,
            )


class _Recorder1(StatsRecorder):
    pass


class _Recorder2(StatsRecorder):
    pass


class _Recorder3(StatsRecorder):
    pass


class RecorderManagerTestCase(TestCase):

    def test_factories(self):
        rm = RecorderManager([
            ('name1', _Recorder1),
            (re.compile(r'.*name(\d+)$'), _Recorder2),
        ])
        rm.add_factory(re.compile(r'.*name(\d+).*'), _Recorder3)

        self.assertIsInstance(rm.get_recorder('name1'), _Recorder1)
        self.assertIsInstance(rm.get_recorder('name2'), _Recorder2)
        self.assertIsInstance(rm.get_recorder('name3_'), _Recorder3)
        self.assertIsInstance(rm.get_recorder('xxx'), NullRecorder)

    def test_record_and_clear(self):
        rm = RecorderManager(default_factory=ConcatRecorder)
        tensors = {
            'name1': [T.random.randn([2, 4]), T.random.randn([3, 4])],
            'name2': [T.random.randn([5]), T.random.randn([6])],
        }
        answers = {key: T.concat(tensors[key], axis=0) for key in tensors}

        def f():
            for name in tensors:
                self.assertIsNone(rm.get(name))
            self.assertEqual(list(rm.iter_all()), [])
            self.assertEqual(rm.get_all(), {})

            for name in tensors:
                for t in tensors[name]:
                    rm.record(name, t)
            for name in tensors:
                assert_allclose(rm.get(name), answers[name])

            def g(records, names):
                if not isinstance(records, dict):
                    records = {k: v for k, v in records}
                self.assertEqual(len(records), len(names))
                self.assertEqual(sorted(names), sorted(records))
                for n in names:
                    assert_allclose(records[n], answers[n])

            g(rm.iter_all(), ['name1', 'name2'])
            g(rm.get_all(), ['name1', 'name2'])

            filter_ = lambda n: n == 'name1'
            g(rm.iter_all(filter_), ['name1'])
            g(rm.get_all(filter_), ['name1'])

        f()
        rm.clear()
        f()

    def test_push_to_stack(self):
        rec1 = RecorderManager({
            'name1': ConcatRecorder,
            'name2': ConcatRecorder,
        })
        rec2 = RecorderManager({
            'name1': ConcatRecorder,
        })
        tensors = {
            'name1': [T.random.randn([2, 4]), T.random.randn([3, 4])],
            'name2': [T.random.randn([5]), T.random.randn([6])],
        }
        answers = {key: T.concat(tensors[key], axis=0) for key in tensors}

        def populate():
            rec1.clear()
            rec2.clear()
            for name in tensors:
                for t in tensors[name]:
                    record_tensor(name, t)

        # test empty
        self.assertFalse(has_recorder_manager())
        populate()
        self.assertEqual(rec1.get_all(), {})
        self.assertEqual(rec2.get_all(), {})

        def g(records, names):
            self.assertEqual(len(records), len(names))
            self.assertEqual(sorted(names), sorted(records))
            for n in names:
                assert_allclose(records[n], answers[n])

        # test one recorder
        self.assertFalse(has_recorder_manager())
        with rec1.push_to_stack() as rm:
            self.assertIs(rm, rec1)
            self.assertTrue(has_recorder_manager())
            populate()
        self.assertFalse(has_recorder_manager())
        g(rec1.get_all(), ['name1', 'name2'])
        g(rec2.get_all(), [])

        # test two recorders
        self.assertFalse(has_recorder_manager())
        with rec1.push_to_stack(), rec2.push_to_stack():
            self.assertTrue(has_recorder_manager())
            populate()
        self.assertFalse(has_recorder_manager())
        g(rec1.get_all(), ['name1', 'name2'])
        g(rec2.get_all(), ['name1'])


class LayerRecorderTestCase(TestCase):

    def test_LayerRecorder(self):
        x = T.random.randn([3, 5])

        # test single output
        linear = tk.layers.Linear(5, 3)
        y = linear(x)
        recorder = with_recorder(linear, 'linear1')
        assert_allclose(recorder(x), y, rtol=1e-4, atol=1e-6)
        with RecorderManager(default_factory=ConcatRecorder).push_to_stack() as rm:
            assert_allclose(recorder(x), y, rtol=1e-4, atol=1e-6)
        assert_allclose(rm.get('linear1.output'), y, rtol=1e-4, atol=1e-6)

        # test multiple outputs
        linear2 = tk.layers.Linear(5, 4)
        branch = tk.layers.Branch([linear, linear2])
        recorder = with_recorder(branch, 'branch')
        y2 = linear2(x)

        with RecorderManager(default_factory=ConcatRecorder).push_to_stack() as rm:
            out1, out2 = recorder(x)
            assert_allclose(out1, y, rtol=1e-4, atol=1e-6)
            assert_allclose(out2, y2, rtol=1e-4, atol=1e-6)

        assert_allclose(rm.get('branch.output.0'), y, rtol=1e-4, atol=1e-6)
        assert_allclose(rm.get('branch.output.1'), y2, rtol=1e-4, atol=1e-6)

    def test_FlowRecorder(self):
        x = T.random.randn([3, 5])

        flow = tk.flows.InvertibleDense(5)
        y, log_det = flow(x)

        recorder = with_recorder(flow, 'flow1')

        for inverse, input_log_det, compute_log_det in product(
                    [True, False],
                    [None, T.random.randn([3])],
                    [True, False]
                ):
            pfx = 'inv_' if inverse else ''

            with RecorderManager(default_factory=ConcatRecorder).push_to_stack() as rm:
                out, out_log_det = recorder(
                    y if inverse else x,
                    input_log_det=input_log_det,
                    inverse=inverse,
                    compute_log_det=compute_log_det
                )

            assert_allclose(out, x if inverse else y, rtol=1e-4, atol=1e-6)
            assert_allclose(rm.get(f'flow1.{pfx}output'), out)

            if compute_log_det:
                tmp = -log_det if inverse else log_det
                assert_allclose(rm.get(f'flow1.{pfx}log_det'), tmp, rtol=1e-4, atol=1e-6)

                if input_log_det is not None:
                    assert_allclose(out_log_det, tmp + input_log_det,
                                    rtol=1e-4, atol=1e-6)
                else:
                    assert_allclose(out_log_det, tmp, rtol=1e-4, atol=1e-6)
                assert_allclose(rm.get(f'flow1.{pfx}output_log_det'), out_log_det,
                                rtol=1e-4, atol=1e-6)
