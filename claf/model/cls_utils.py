import csv
from collections import defaultdict

from seqeval.metrics.sequence_labeling import get_entities


# pycm
def write_confusion_matrix_to_csv(file_path, pycm_obj):
    with open(file_path + ".csv", "w") as f:
        indicator = "target/predict"

        fieldnames = [indicator] + pycm_obj.classes + ["FN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        data = dict(pycm_obj.matrix)
        FN = dict(pycm_obj.FN)

        for row_idx in fieldnames[1:-1]:  # remove indicator and FN
            row = {indicator: row_idx}
            row.update(
                {
                    col_idx: data[row_idx][col_idx]
                    for col_idx in data[row_idx]
                    if col_idx in fieldnames
                }
            )
            row.update({"FN": FN[row_idx]})
            writer.writerow(row)

        row = {indicator: "FP"}
        row.update(dict(pycm_obj.FP))
        writer.writerow(row)


def get_tag_entities(sequence, tag_texts):
    """
    Make an IOB tag list into an entity dictionary

    ex) sequence: "looking for a moderate priced viet nam or thai restaurant"
        tag_texts: ["O", "O", "O", "B-price", "O", "B-food", "I-food", "O", "B-food", "O"]

        => {
            "price": ["moderate"],
            "restaurant_type": ["viet nam", "thai"]
        }

    * Args:
        sequence: a sequence, with the number of its tokens
            obtained from splitting by whitespace is the same with the number of elements in tag_texts
        tag_texts: a list of tags in their text form

    * Returns:
        tag_entities: dictionary consisting of
            - $tag_name: list of entities
    """

    words = sequence.split()
    entities = get_entities(tag_texts)  # seqeval

    slots = defaultdict(list)
    for slot, start_idx, end_idx in entities:
        value = " ".join(words[start_idx : end_idx + 1])

        if value:
            slots[slot].append(value)

    return dict(slots)
