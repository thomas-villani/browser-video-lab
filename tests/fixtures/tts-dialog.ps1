# A three-voice conversation with Windows TTS, for the speakers layer: each turn spoken to its
# own WAV (tests/fixtures/dialog/NN-<voice>.wav), joined by scripts/tts-dialog.mjs into
# tests/fixtures/tts-dialog.mp4 with the exact turns (tts-dialog.rttm). PowerShell 7 (Windows
# PowerShell 5 cannot select the OneCore voice Mark):
#   pwsh -File tests/fixtures/tts-dialog.ps1 <out dir>
param([string]$Out)
Add-Type -AssemblyName System.Speech
$turns = @(
  @('Microsoft David Desktop', 'Welcome back to the workshop. Today I am joined by two guests who restore old radios, and we will talk about where to start.'),
  @('Microsoft Zira Desktop', 'Thanks for having me. The first thing I tell people is to never plug in an old radio before checking the capacitors.'),
  @('Microsoft Mark', 'I agree completely. A dried out capacitor can take the whole power supply with it, and those parts are hard to find.'),
  @('Microsoft David Desktop', 'So what tools does a beginner actually need?'),
  @('Microsoft Zira Desktop', 'A decent soldering iron, a multimeter, and patience. Most of the work is cleaning and testing, not soldering.'),
  @('Microsoft Mark', 'And a variac, if you can find one. It lets you bring the voltage up slowly and watch for smoke.'),
  @('Microsoft David Desktop', 'Watch for smoke. That sounds like good advice for most of life.'),
  @('Microsoft Zira Desktop', 'It really is. We will put a list of parts in the show notes.')
)
New-Item -ItemType Directory -Force $Out | Out-Null
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$i = 0
foreach ($t in $turns) {
  $s.SelectVoice($t[0])
  $name = '{0:D2}-{1}.wav' -f $i, ($t[0] -replace 'Microsoft | Desktop', '')
  $s.SetOutputToWaveFile((Join-Path $Out $name))
  $s.Speak($t[1])
  $s.SetOutputToNull()
  $i++
}
Write-Host "turns: $i"
