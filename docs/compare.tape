# VHS tape: the detector's money moment — real captures, real verdict.
# Prereq: ref.json (26B 4-bit) and cand.json (same model, 2-bit) produced by
# `clausius capture` in the working dir below.
# Render: vhs docs/compare.tape  (from the clausius repo root)
Output docs/compare.gif
Set FontSize 15
Set Width 1000
Set Height 470
Set Theme "Catppuccin Mocha"
Set Padding 12
Set TypingSpeed 40ms

Hide
Type "cd /private/tmp/claude-503/-Users-beata-Developer-clausius/e6ec2640-e206-49c7-ac6d-b7e02f43892b/scratchpad/clausius_demo && source /Users/beata/Developer/clausius/.venv/bin/activate && clear"
Enter
Sleep 1s
Show

Type "# ref.json: 26B MoE at 4-bit — cand.json: the same model at 2-bit"
Enter
Type "clausius compare ref.json cand.json --show 1"
Sleep 600ms
Enter
Sleep 4s
Type "# no labels, 60 prompts — the 2-bit config measures 73 points lower on instruction adherence"
Sleep 3s
