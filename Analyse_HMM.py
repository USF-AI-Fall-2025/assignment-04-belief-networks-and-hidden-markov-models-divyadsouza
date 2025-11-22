import HMM


def map_misspelled_to_correct(lines):
    typo_mappings = {}
    valid_words = []
    for line in lines:
        if ":" not in line:
            continue
        correct, incorrect = line.strip().lower().split(":", 1)
        typos = incorrect.split()
        for typo in typos:
            typo_mappings[typo] = correct
        valid_words.append(correct)

    return typo_mappings, valid_words


def test_HMM(file: str):

    lines = HMM.get_spellings(file=file)
    emissions = HMM.compute_emission_probabilities(lines)
    transitions = HMM.compute_transition_probabilities(lines)
    typo_mappings, valid_words = map_misspelled_to_correct(lines=lines)

    correct = {}
    incorrect_but_valid = {}
    incorrect = {}
    for typo, expected in typo_mappings.items():
        actual = HMM.correct_text(typo, emissions, transitions)
        if actual == expected:
            correct[typo] = actual
        elif actual in valid_words:
            incorrect_but_valid[typo] = actual
        else:
            incorrect[typo] = actual
    return correct, incorrect_but_valid, incorrect


if __name__ == "__main__":

    correct, incorrect_but_valid, incorrect = test_HMM("aspell.txt")
    print("Words correctly corrected")
    print(correct, "\n")

    print("Words corrected to valid words")
    print(incorrect_but_valid, "\n")

    print("Words not corrected")
    print(incorrect)
