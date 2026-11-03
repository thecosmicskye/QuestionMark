# ?

<img src="logo.png" alt="?" width="150" align="right">

**?** is a Rocket League bot for [RLBot v5](https://rlbot.org) by Cosmic Skye: a neural network with its own policy
for each of 1v1, 2v2 and 3v3, driving the Hybrid. It runs on the CPU with ONNX Runtime.

- **Steps:** 1.62 trillion (3,420 years of driving)
- **Model:** a transformer, 4,083,585 parameters per policy: one 256-wide token each for the cars, the ball and the
  game state; two attention layers with four heads; a 2048-wide MLP
- **Actions:** every 8 ticks (15 times a second) one of 4,992: 624 controller inputs times a start delay of 0 to 7 ticks
- **Training:** self-play against a league of its past versions, rewarded only for goals scored and conceded; about
  16 days on up to 65,536 simulated arenas at once. Opponents drove a mix of cars in the proportions seen in ranked play

## Run from source

```
pip install -r requirements.txt
```

Then add this folder in the RLBot GUI and put "?" on a team. It has a policy each for 1v1, 2v2 and 3v3; any other matchup plays the policy for the bigger team (2v1 plays 2v2, 4v4 plays 3v3).

## Files

| file | |
|---|---|
| `bot.py` | the RLBot controller: runs the policy's programs and holds each car's controls until its next command |
| `packet.py` | copies packet fields into the tables the policy reads |
| `policy/` | the policy: `manifest.json`, the state step (`step.onnx`) and one program per matchup |
| `convert_policy.py` | makes `policy/` from a `policy.pt2` archive and checks it against the archive |

## License

Copyright © 2026 Cosmic Skye, under the [GNU AGPL v3.0](LICENSE) (code and policy). Rocket League is a trademark of
Psyonix; this project is not affiliated with or endorsed by Psyonix or Epic Games.
