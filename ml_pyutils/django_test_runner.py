import importlib
import os
import time
import unittest

from django.test.runner import DiscoverRunner


class TimedTextResult(unittest.TextTestResult):
    def startTest(self, test):
        self._test_start_time = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = time.perf_counter() - self._test_start_time
        self.durations = getattr(self, "durations", {})
        self.durations[test.id()] = elapsed
        super().stopTest(test)

    def _test_location(self, test) -> str:
        mod_name = test.__class__.__module__
        cls_name = test.__class__.__qualname__
        meth_name = getattr(test, "_testMethodName", None)

        file_path = None
        try:
            mod = importlib.import_module(mod_name)
            file_path = getattr(mod, "__file__", None)
        except Exception:
            pass

        # Normalize .pyc -> .py when possible
        if file_path and file_path.endswith(".pyc"):
            maybe_py = file_path[:-1]
            if os.path.exists(maybe_py):
                file_path = maybe_py

        dotted = f"{mod_name}.{cls_name}"
        if meth_name:
            dotted += f".{meth_name}"

        return (
            f"\n\t📍 file: {file_path}\n\t📦 module: {dotted})"
            if file_path
            else f"\n\tmodule:{dotted}"
        )

    def printErrorList(self, flavour, errors):
        # This is used by unittest to print both FAIL and ERROR sections.
        for i, (test, err) in enumerate(errors):
            error_flavour = f"{flavour} ({i + 1} / {len(errors)})"

            test_location_string = self._test_location(test)
            self.stream.writeln(f"❌ {error_flavour}")
            self.stream.writeln(test_location_string)
            self.stream.writeln()
            self.stream.writeln("🐞 " + err)
            self.stream.writeln(test_location_string)
            self.stream.writeln()
            self.stream.writeln(f"🔚 {error_flavour}")
            self.stream.writeln(self.separator1)


class TimedTextRunner(unittest.TextTestRunner):
    resultclass = TimedTextResult
    max_num_prints = 5

    def run(self, test):
        result = super().run(test)
        durations = getattr(result, "durations", {})
        if durations:
            print("\nTest durations:")
            for test_id, secs in sorted(durations.items(), key=lambda x: x[1], reverse=True)[
                : self.max_num_prints
            ]:
                print(f"{secs:8.4f}s  {test_id}")
            print(f"\nTotal: {sum(durations.values()):.4f}s")
        return result


class TimedDiscoverRunner(DiscoverRunner):
    def run_suite(self, suite, **kwargs):
        return TimedTextRunner(
            verbosity=self.verbosity,
            failfast=self.failfast,
            buffer=self.buffer,
        ).run(suite)
