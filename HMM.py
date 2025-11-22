
# I used AI to figure out my algorithm to calculate emission and transition counts
# to cleanly perform emission_counts['a']['b'] += 1
from collections import defaultdict, Counter

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
START = "<START>"
END = "<END>"


def compute_emission_probabilities(lines):
    # emission_counts[correct_letter][typed_letter] = count
    emission_counts = defaultdict(Counter)
    total_counts = Counter()

    for line in lines:
        if ':' not in line:
            continue
        correct, typos_str = line.strip().lower().split(':', 1)
        correct = correct.strip()
        typos = [typo.strip()
                 for typo in typos_str.split() if typo.strip()]

        for typo in typos:
            # zip() stops when the shorter word ends
            for c, t in zip(correct, typo):
                if c.isalpha() and t.isalpha():
                    emission_counts[c][t] += 1
                    total_counts[c] += 1

    # shape: 'a': {'a': 0.95, 's': 0.03, 'q': 0.02}
    emission_probs = {}
    for c in emission_counts:
        total = sum(emission_counts[c].values())
        emission_probs[c] = {t: count / total for t,
                             count in emission_counts[c].items()}

    return emission_probs


def compute_transition_probabilities(lines):
    trans_counts = defaultdict(Counter)

    for line in lines:
        if ":" not in line:
            continue
        correct, _ = line.strip().lower().split(":", 1)
        letters = [ch for ch in correct.strip() if ch.isalpha()]
        if not letters:
            continue

        trans_counts[START][letters[0]] += 1
        for a, b in zip(letters, letters[1:]):
            trans_counts[a][b] += 1
        trans_counts[letters[-1]][END] += 1

    transitions = {}

    start_total = sum(trans_counts[START].values())
    transitions[START] = {}
    if start_total > 0:
        for c in ALPHABET:
            count = trans_counts[START][c]
            if count > 0:
                transitions[START][c] = count / start_total

    # From letters: letters + END
    for s in ALPHABET:
        total = sum(trans_counts[s][c]
                    for c in ALPHABET) + trans_counts[s][END]
        if total == 0:
            continue
        transitions[s] = {}
        for c in ALPHABET:
            if trans_counts[s][c] > 0:
                transitions[s][c] = trans_counts[s][c] / total
        if trans_counts[s][END] > 0:
            transitions[s][END] = trans_counts[s][END] / total

    return transitions


def viterbi_decode_word(word, E, T, epsilon=1e-12):
    """
    - E[s][o] = P(observation=o | state=s)
    - T[a][b] = P(next_state=b | state=a)
    """
    observations = [ch for ch in word.lower() if ch.isalpha()]
    if not observations:
        return word.lower()

    states = [START] + ALPHABET + [END]
    state_values = ALPHABET

    def p_emit(s, o):
        return E.get(s, {}).get(o, 0.0) or epsilon

    def p_trans(a, b):
        return T.get(a, {}).get(b, 0.0) or epsilon

    # M[t,s] = 0 ; Backpointers[t,s] = 0
    M = []
    Backpointers = []
    for t in range(len(observations)):
        row = {}
        b_row = {}
        for s in state_values:
            row[s] = 0.0
            b_row[s] = 0
        M.append(row)
        Backpointers.append(b_row)

    # M[1,s] = T[states[0], s] * E[s, O[1]]
    # (zero-based indices) → M[0][s] = T[START, s] * E[s, observations[0]]
    for s in state_values:
        M[0][s] = p_trans(states[0], s) * p_emit(s, observations[0])

    # For o in observations (time recursion)
    for o in range(1, len(observations)):
        for s in state_values:
            # val = max( M[state2, o-1] * T[state2, s] * p(s, o) ) over state2
            best_val = 0.0
            best_prev = 0
            emit = p_emit(s, observations[o])
            for state2 in state_values:
                cand = M[o-1][state2] * p_trans(state2, s) * emit
                if cand > best_val:
                    best_val = cand
                    best_prev = state2
            M[o][s] = best_val
            Backpointers[o][s] = best_prev

    # Best = max(M[t,:])
    last_t = len(observations) - 1
    Best = 0
    Best_val = 0.0
    for s in state_values:
        cand = M[last_t][s] * p_trans(s, END)
        if cand > Best_val:
            Best_val = cand
            Best = s

    # Backtrack:
    # list = [] ; for o in observations.reverse: list.push(Best); Best = Backpointers[Best, o]
    path = []
    o = last_t
    while o >= 0:
        path.append(Best)
        Best = Backpointers[o][Best] if o > 0 else Best
        o -= 1
    path.reverse()

    return "".join(path)


def correct_text(text, emission_probs, transition_probs):
    words = text.split()
    fixed_words = []
    for w in words:
        decoded = viterbi_decode_word(w, emission_probs, transition_probs)
        fixed_words.append(decoded)
    return " ".join(fixed_words)


def get_spellings(file):
    aspell_path = file

    with open(aspell_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    return lines


if __name__ == "__main__":

    lines = get_spellings("aspell.txt")
    # Compute emission and transition probabilities using the same data
    emissions = compute_emission_probabilities(lines)
    transitions = compute_transition_probabilities(lines)

    # I used AI for elegant formatting
    # Print emission probabilities
    '''
    print("--- EMISSIONS ---")
    for correct_letter, typed_dict in sorted(emissions.items()):
        sorted_typed = sorted(typed_dict.items(),
                              key=lambda x: x[1], reverse=True)
        probs_str = " ".join([f"{t}:{p:.3f}" for t, p in sorted_typed])
        print(f"{correct_letter} -> {probs_str}")

    # Print transition probabilities
    print("\n--- TRANSITIONS ---")
    if START in transitions:
        start_row = sorted(
            transitions[START].items(), key=lambda x: x[1], reverse=True)
        print(f"{START} -> " + " ".join(f"{n}:{p:.3f}" for n, p in start_row))

    for s in ALPHABET:
        if s not in transitions:
            continue
        row = sorted(transitions[s].items(), key=lambda x: x[1], reverse=True)
        row_str = " ".join(f"{n}:{p:.3f}" for n, p in row)
        print(f"{s} -> {row_str}")
        '''
    try:
        while True:
            user_text = input(
                "Type text to correct (keyboard interrupt to exit): ").strip()
            if not user_text:
                print("")
                continue
            print(correct_text(user_text, emissions, transitions))
    except (EOFError, KeyboardInterrupt):
        pass
