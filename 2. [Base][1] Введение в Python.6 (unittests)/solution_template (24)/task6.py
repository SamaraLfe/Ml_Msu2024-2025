def check(s, filename):
    words = s.lower().split()
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    sorted_words = sorted(word_count.keys())

    with open(filename, 'w') as file:
        for word in sorted_words:
            file.write(f"{word} {word_count[word]}\n")