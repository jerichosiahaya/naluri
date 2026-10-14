# Encoder and decoder decision models: technical architecture

Written in ASD-STE100 Simplified Technical English. Status date: 2026-10-14.

## 1. Scope

This document gives the architecture of two types of System One decision model:

- The encoder decision model. Naluri uses this type.
- The decoder decision model. Kev, Jev and the leaders of System One Bench use this type.

This document also gives the test data for the two types and a procedure to select a type.

## 2. Terms

| Term | Meaning |
|---|---|
| State | The input data that the model reads. The state is text, or named fields. |
| Question | The decision that the model makes about the state. |
| Option | One possible answer to a question. |
| Question type | `noul` (yes or no), `choice` (one option from a list) or `score` (one level on a scale). |
| Marker | The token position where the model reads the vector for one option. |
| Calibration | The agreement between the probability that the model gives and the true accuracy. |
| ECE | Expected calibration error. A low ECE shows good calibration. |
| Cascade | Two models in series. The second model gets only the cases that the first model cannot decide with sufficient confidence. |

## 3. Function of a decision model

A decision model gets one state and one or more questions. For each question, the model gives a probability for each option. The sum of the probabilities for one question is 1.

The decision model does not write text. It only selects options. Thus, the model must be fast and its probabilities must be correct.

## 4. Encoder decision model

### 4.1 Components

| Component | Naluri value |
|---|---|
| Encoder | XLM-RoBERTa base (280M parameters, 12 layers) or mmBERT-base (307M parameters, 22 layers) |
| Attention | Bidirectional. Each token can see all other tokens. |
| Decision head | LayerNorm, Linear, GELU, Linear. The head gives one score for each marker. |
| Output | Softmax over the option scores of one question |
| Calibration | One temperature for each question type, fitted on validation data |

### 4.2 Input layout

The model puts the question, the options and the state into one sequence:

```
<s> {type} question: {instructions} </s> <mask> {option 1} <mask> {option 2} ... </s> {state} </s>
```

Each option starts with a `<mask>` token. This token is the marker for that option.

Token limits (refer to `s1.py`):

| Item | Limit |
|---|---|
| Full sequence | 512 tokens (XLM-R). mmBERT can accept 8,192 tokens. |
| Question and options | 192 tokens |
| One option | 48 tokens |
| State | The tokens that remain |

If there are many options, the model makes each option shorter. Thus, a question with many options loses text.

### 4.3 Data flow

1. The tokenizer makes the sequence.
2. The encoder makes one vector for each token. Because the attention is bidirectional, each marker vector contains data from the question, from all options and from the state.
3. The decision head reads the vector at each marker and gives one score.
4. The model divides the scores by the temperature for the question type.
5. The softmax gives the probability for each option.

The model does all steps in one forward pass. The model does not generate tokens.

### 4.4 Special conditions for mmBERT

mmBERT uses local attention in most layers. The window is 128 tokens. Each third layer uses global attention. Thus, a marker gets data from a far part of the state only through the global layers.

## 5. Decoder decision model

### 5.1 Components

| Component | Typical value |
|---|---|
| Decoder | A language model with 1B to 27B parameters (for example, Qwen3-4B) |
| Attention | Causal. Each token can see only the tokens before it. |
| Adaptation | LoRA adapters on a frozen base model, or a full fine-tune |
| Scoring | Method A or method B (refer to 5.3) |
| Calibration | Temperature, fitted on validation data |

### 5.2 Input layout

Because the attention is causal, the encoder layout is not correct for a decoder. In the encoder layout, the options come before the state. Thus, in a decoder, the options cannot see the state.

Use this layout for a decoder:

```
{state} {question} {option 1} {option 2} ... {last token}
```

The state comes first. The options come last, so each option can see the state and the question.

### 5.3 Scoring methods

Method A, label probability:

1. Give each option a short label (for example, `A`, `B`, `C`).
2. Read the next-token probability of each label at the last position.
3. Use these probabilities as the option scores.

Method B, marker vector:

1. Put a marker token after each option.
2. Read the vector at each marker.
3. Use a small head to give one score for each marker.

In method B, an option cannot see the options that come after it. Method A does not have this problem, because the last position can see all options.

### 5.4 State cache

A decoder can keep the key and value cache of the state. If the model must answer many questions about one state, the model computes the state only one time. Each subsequent question uses the cache. The encoder cannot do this, because each question changes the full sequence.

## 6. Comparison

| Property | Encoder | Decoder |
|---|---|---|
| Parameters | 140M to 600M | 1B to 27B |
| Latency, GPU | Approximately 5 ms (Naluri, median) | Tens to hundreds of ms |
| Latency, CPU | 50 ms to 60 ms (Naluri) | Not practical |
| System One Bench index | 58 to 60 | 71 to 83 |
| World knowledge | Low | High |
| Many options | Limited by the 192-token budget | Better |
| Long state | 512 tokens (XLM-R), 8,192 tokens (mmBERT) | 32,000 tokens or more |
| Calibration | Good after temperature fit (Naluri M9 ECE: 0.027) | Must also have a temperature fit |
| Many questions about one state | One pass for each question | One state pass, then a cache |
| Training hardware | One 8 GB laptop GPU | Larger GPU |
| Training method | Full fine-tune | LoRA, usually |

## 7. Test data

### 7.1 System One Bench (index = mean suite accuracy)

| Model | Type | Index |
|---|---|---|
| Perplexity Decider V1 27B | Decoder | 83.3 |
| Kev 4B | Decoder | 76.8 |
| Together Tev1 4B | Decoder | 71.5 |
| Naluri, mmBERT-base | Encoder | 61.0 |
| Laya Router | Encoder | 60.4 |
| Naluri M9 (XLM-R base) | Encoder | 58.0 |

All models with an index of more than 70 are decoders.

### 7.2 Naluri encoder backbones (gen_dev macro accuracy)

The gen_dev set contains task types that are not in the training data.

| Model | Training | gen_dev |
|---|---|---|
| Naluri M9, XLM-R base | Three training rounds (M6, M8, M9) | 0.652 |
| Naluri, mmBERT-base, epoch 4 of 5 | One run on the M9 data, from the base model | 0.674 |

### 7.3 Naluri encoder backbones (System One Bench)

| Property | M9 (XLM-R base) | mmBERT-base |
|---|---|---|
| Index | 58.0 | 61.0 |
| ECE | 0.027 | 0.057 |
| Accuracy on the 50% most confident answers | 67.8% | 70.6% |
| Latency, GPU, median | 4.9 ms | 9.2 ms |

mmBERT is better on NLI (XNLI English: 62.7% to 76.0%), emotion (70.0% to 84.5%), intent (52.5% to 65.5%) and typed decisions (46.9% to 54.6%). mmBERT is worse on sentiment (SST-2: 74.5% to 68.5%, SST-5: 36.5% to 28.0%) and BoolQ (60.0% to 55.0%).

mmBERT has 22 layers and XLM-R base has 12 layers. Thus, mmBERT is approximately two times slower.

Result: a newer encoder gives an increase of 3 points. It does not close the gap to the decoders.

## 8. Cascade architecture

The cascade uses the two types together:

1. The encoder answers each question.
2. If the highest probability is more than the threshold, the system uses the encoder answer.
3. If the highest probability is less than the threshold, the system sends the question to the decoder.
4. The decoder answers the question.

The cascade is safe only if the encoder calibration is good. Naluri has an ECE of 0.027, so its probabilities are applicable for the threshold.

Before you use the cascade, measure the coverage curve. For each threshold, record:

- The percentage of questions that the encoder answers.
- The accuracy of the encoder on these questions.

## 9. Procedure to select a model type

1. Measure the latency limit of the application.
2. If the limit is less than 20 ms, or the system must operate on a CPU, use an encoder.
3. If the questions need world knowledge or reasoning, use a decoder.
4. If a question has more than 12 options, use a decoder.
5. If the state is longer than 512 tokens, use a decoder or a long-context encoder (mmBERT).
6. If the application has a high volume and also has difficult questions, use the cascade (refer to section 8).
7. For all types, fit the temperatures on validation data before you use the probabilities.
