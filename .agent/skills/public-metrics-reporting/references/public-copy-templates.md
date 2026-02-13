# Public Copy Templates

Use these templates after collecting `before`, `after`, and `delta` values.

## TLDR (One Line)

```text
TLDR: False alarms moved from <before_fpr>% to <after_fpr>% (<delta_fpr> pts), scam catch rate moved from <before_recall>% to <after_recall>% (<delta_recall> pts), model size is <size_kb> KB.
```

## Tweet (Mainstream, <=280 chars)

```text
Janitr update: scam detection now catches <after_recall>% of scams (was <before_recall>%) with <after_fpr>% false alarms (was <before_fpr>%). Precision is <after_precision>% (<before_precision>% before). Model size: <size_kb> KB. <short_tradeoff_sentence>
```

## Release Notes (3-6 Bullets)

```text
- Scam precision: <before_precision>% -> <after_precision>% (<delta_precision> pts)
- Scam recall: <before_recall>% -> <after_recall>% (<delta_recall> pts)
- False alarm rate (scam): <before_fpr>% -> <after_fpr>% (<delta_fpr> pts; lower is better)
- Topic recall: <before_topic_recall>% -> <after_topic_recall>% (<delta_topic_recall> pts)
- Model size: <before_size> -> <after_size>
- Tradeoff: <one sentence about what got worse, if anything>
```

## Plain-Language Glossary

- `precision`: When Janitr flags something, how often it is right.
- `recall`: How many real cases Janitr catches.
- `false positive rate`: How often Janitr raises a false alarm.

## Safety Checklist

- Include both `before` and `after` for every claim.
- Mention at least one tradeoff when any metric regresses.
- Avoid unexplained terms like `holdout`, `calib`, `macro`, and `micro`.
- Keep percentages rounded to one decimal place for public copy.
