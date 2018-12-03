# Text Processing - Document Retrieval Assignment

Date Submitted: Saturday, 10 November 2018 03:32:32 o'clock GMT

- Implementation (out of 15): **11**
- Report (out of 10): **9**
- Total: **20**

# Comments:

Scores are all as we would expect. Run time is quite good, but could be slightly improved.

You correctly calculate the document collection size in the constructor of the class.

The main hinderance on performance in your solution is that you are calculating the document vector size
each iteration and for each query. We could calculate these values in the constructor of the class.

Report is very good and presented nicely. You provide a good discussion on your implementation and results.
You justify correctly why the stop list is not as effective on the tfidf weighting.

### RESULTS FROM AUTOMATED TESTING
------------------------------------------------------------------
- F-SCORES:

|        | stp-/stm- | stp+/stm- | stp-/stm+ | stp+/stm+ |
|:------:|:---------:|:---------:|:---------:|:---------:|
|  TFIDF |    0.18   |    0.19   |    0.23   |    0.24   |
|   TF   |    0.07   |    0.14   |    0.10   |    0.18   |
| Binary |    0.06   |    0.12   |    0.08   |    0.15   |
------------------------------------------------------------------

- TIMES:

|        | stp-/stm- | stp+/stm- | stp-/stm+ | stp+/stm+ |
|:------:|:---------:|:---------:|:---------:|:---------:|
|  TFIDF |    7.72   |    5.90   |    4.78   |    3.33   |
|   TF   |    6.85   |    6.26   |    3.67   |    2.96   |
| Binary |    6.75   |    5.75   |    3.59   |    3.29   |
------------------------------------------------------------------

- PRECISION:

|        | stp-/stm- | stp+/stm- | stp-/stm+ | stp+/stm+ |
|:------:|:---------:|:---------:|:---------:|:---------:|
|  TFIDF |    0.21   |    0.22   |    0.26   |    0.27   |
|   TF   |    0.08   |    0.16   |    0.11   |    0.20   |
| Binary |    0.07   |    0.13   |    0.09   |    0.16   |
------------------------------------------------------------------

- RECALL:

|        | stp-/stm- | stp+/stm- | stp-/stm+ | stp+/stm+ |
|:------:|:---------:|:---------:|:---------:|:---------:|
|  TFIDF |    0.17   |    0.18   |    0.21   |    0.22   |
|   TF   |    0.06   |    0.13   |    0.09   |    0.16   |
| Binary |    0.06   |    0.10   |    0.07   |    0.13   |
------------------------------------------------------------------
[. -> timeout (30s)]
[x -> code crashed]