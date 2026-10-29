# Speak a script to a WAV with Windows TTS and record each word's start (ground truth for word
# timestamps). How tts-words.mp4 was made (Windows PowerShell, for System.Speech):
#   powershell -File tests/fixtures/tts.ps1 words.wav tests/fixtures/tts-words.tsv tests/fixtures/tts-words.txt
#   ffmpeg -f lavfi -i color=gray:s=640x360:r=25 -i words.wav -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac tests/fixtures/tts-words.mp4
# SAPI's AudioPosition counts at 16 kHz while writing 22.05 kHz: true time = tsv time x 16000/22050.
param([string]$Wav, [string]$Json, [string]$Script)
Add-Type -AssemblyName System.Speech
Add-Type -ReferencedAssemblies System.Speech -TypeDefinition @'
using System.Collections.Generic;
using System.Speech.Synthesis;
public static class Tts {
  public static List<string> Run(string text, string wav) {
    var s = new SpeechSynthesizer();
    var words = new List<string>();
    s.SpeakProgress += (o, e) => words.Add(e.AudioPosition.TotalSeconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture) + "\t" + e.Text);
    s.SetOutputToWaveFile(wav);
    s.Speak(text);
    s.SetOutputToNull();
    return words;
  }
}
'@
$words = [Tts]::Run((Get-Content -Raw $Script), $Wav)
$words | Set-Content -Encoding utf8 $Json
Write-Host "words: $($words.Count)"
