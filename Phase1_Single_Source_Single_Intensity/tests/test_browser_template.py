"""Catch executable-template syntax failures before serving the explorer."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from rind_phase1.data import PROJECT_ROOT


class BrowserTemplateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is optional; used only to parse browser JavaScript')
    def test_javascript_parses(self):
        template=(PROJECT_ROOT/'src/rind_phase1/templates/results.html').read_text()
        script=template.split('<script>')[1].split('</script>')[0]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'browser.js';path.write_text(script)
            result=subprocess.run(['node','--check',str(path)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
