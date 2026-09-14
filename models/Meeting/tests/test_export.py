import unittest
from paiton_meeting.export import export, clock

class ExportTests(unittest.TestCase):
    def test_carry_and_long_timestamps(self):
        self.assertEqual(clock(3599.9999), '01:00:00.000')
        self.assertEqual(clock(26*3600), '26:00:00.000')

    def test_subtitle_escaping_and_speaker_rename(self):
        result={'segments':[dict(id='s1',start=1.25,end=2.5,speaker='SPEAKER_00',text='<script>x</script> & hello')]}
        rendered=export(result,'vtt',{'SPEAKER_00':'Alice'})
        self.assertIn('00:00:01.250 --> 00:00:02.500', rendered)
        self.assertIn('Alice: &lt;script&gt;x&lt;/script&gt; &amp; hello', rendered)
        self.assertNotIn('<script>', rendered)
        self.assertIn('00:00:01,250',export(result,'srt'))

    def test_summary_requires_available_evidence(self):
        with self.assertRaises(ValueError):export({},'summary.txt')
        with self.assertRaises(ValueError):export({},'html')
