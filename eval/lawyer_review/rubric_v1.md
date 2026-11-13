# Lawyer review rubric, v1

This rubric is for a practising lawyer reviewing a sample of Nyaya Search
answers. The review is qualitative evidence about whether the answers are
useful. It is **never a gate**: no goal passes or fails on it.

## What you receive

`nyaya-eval lawyer-export` writes two files for one answers run:

*   `review.html`: the questions and answers, readable in any browser. Each
    answer sentence shows the case, the paragraphs and the exact quote it
    relies on. Statements that the passages do not cover a point ("gaps") are
    listed after the answer.
*   `ratings.csv`: one row per answer, with empty rating columns for you to
    fill in (in a spreadsheet program, then saved as CSV).

Please judge each answer as a starting point for legal research on the
question asked, using your own knowledge and, where needed, the cited
judgments.

## The four questions

Rate each question 1, 2 or 3:

| Rating | Meaning |
| --- | --- |
| 1 | No |
| 2 | Partly |
| 3 | Yes |

1.  **Right paragraphs cited** (`right_paragraphs`): do the cited paragraphs
    actually say what the sentence relies on them for, and are they the
    paragraphs you would cite for that point?
2.  **Law stated correctly** (`law_correct`): is the law stated correctly,
    including its conditions and exceptions, and attributed to the Court
    (not to a party's argument, a dissent or an overruled view)?
3.  **Nothing important missing** (`nothing_missing`): does the answer cover
    what a researcher would need to know first on this question, without a
    significant omission?
4.  **Usable as a research starting point** (`usable`): would you use this
    answer, with its citations, as the starting point for research on the
    question?

Leave a rating blank if you cannot judge it; blank ratings are left out of
the averages. Use `comment` for anything else (for example a better
authority, an overruling, or a misleading sentence). Put your name or
initials in `reviewer`.

## Filling in `ratings.csv`

*   Do not change `review_id`, `question_id` or the order of the columns.
*   Each rating is `1`, `2`, `3` or blank. Any other value is rejected on
    import, with the row named.
*   Return the file; it is read with `nyaya-eval lawyer-import`.
