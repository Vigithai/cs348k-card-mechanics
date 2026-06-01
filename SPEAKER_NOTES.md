# Speaker Notes — Learning to Play Balatro

---

## Slide 1 · Title

Hi, I'm [name]. This project is about using RL to get better at Balatro — a roguelike card game — by building a bot that can play it, and then reading what the bot learned.

The headline number is 35% win rate for the RL agent versus 17% for the best rule-based bot I wrote. That 2× gap is what we'll unpack.

---

## Slide 2 · The Game

[90 seconds — live demo or point to screenshot]

Balatro is basically: you're dealt 7 cards, you play poker hands to score chips, and you need to hit a chip target before you run out of hands. If you clear the target, you advance to the next blind with a higher target. Three blinds make an ante.

The scoring table on screen shows why the hand choice matters: a pair scores about 20 chips, a flush scores 136+, a four of a kind can score 728. The gap between premium hands and weak hands is massive.

My project models just the card mechanics — no jokers, no shop — to isolate the strategy question: what hands do you build toward, and when do you commit to hunting one?

---

## Slide 3 · Performance

The three bars are: me (~50% on ante 1), the best scripted bot (17%), and the RL agent (35%).

A few important caveats: my number is a rough estimate from playing the real game. The bots run on a simplified environment. So this is directional, not a direct comparison. The point is: the RL agent got meaningfully closer to human-level play than any of the scripted bots, without any hand-crafted strategy.

---

## Slide 4 · Questions

Before I show you the results, here are the four questions I wanted to answer from the data. These motivated the whole analysis pipeline.

Q1 through Q4 — read them off, brief pause. The key framing: "discarding is a bet — when is that bet rational?" and "what's the one rule a human can actually use?"

---

## Slide 5 · Analysis pipeline

Here's how I got from games to insights.

I ran 500 games each with the RL bot and the best scripted bot. I only kept winning games — 177 RL wins and 84 scripted wins. Each saved trace records every action: the hand before, the cards played or discarded, chips gained.

The key trick for detecting intent: count consecutive discards. Two or more discards in a row means the bot is hunting something. The hand type played immediately after tells you what it was building toward — straight, flush, full house.

---

## Slide 6 · Q1 — Setup then strike

The chart shows the most common two-play sequences in winning games, comparing RL (gold) vs the scripted bot (blue).

The RL bot's top sequence is two_pair → flush, 104 times. It banks safe chips with a two_pair first, then hunts and lands a flush to close the blind. The scripted bot's top sequence is pair → pair — it just grinds the same safe hand repeatedly.

This is the clearest evidence that RL discovered a deliberate strategy: set up your margin, then bet big.

---

## Slide 7 · Q2 — Hunting is principled

Left chart: what does the bot hunt for after 2+ discards? Flush and straight dominate. Right chart: does chip pressure affect the hunt's success rate?

The surprising answer: barely. When comfortable (>50% of chips needed still remaining) the bot hits a premium hand 57% of the time after a hunt. When desperate it's 56%. The success rate barely moves.

This tells us the bot isn't panic-discarding — it hunts when it sees the cards, not when it feels pressure.

---

## Slide 8 · Q3 — RL is more selective

Three comparisons: win margin, discard rate, premium hand rate.

RL discards less (43% of actions) and plays fewer premium hands (45%) than the scripted bot (59% discards, 66% premium). But RL wins with *more* margin (+120 vs +107).

The scripted bot was built with explicit flush-hunting bias — it discards aggressively toward flushes even when it shouldn't. RL learned restraint: it knows when a two_pair is good enough and doesn't over-hunt.

---

## Slide 9 · Q4 — The rule

The takeaway a human can actually use: within RL's winning games, the aggressive ones (more than 1 discard per blind on average) cleared with 25% more margin (+121 vs +97) and played more premium hands.

Hunting is worth it. If you have the discards and you see the cards for a flush or straight, burn the discards and go for it. The bot validated this across 177 winning games.

---

## Wrap-up / Q&A prompts

- "The next step is adding jokers and the shop layer — that's where Balatro's real complexity lives."
- "The RL training was only 500 episodes; with more compute the gap would likely grow."
- "The simplified environment is a limitation — real Balatro has variance from deck RNG and joker interactions that we don't model."
