Event_types = set()
Arguments = set()
Roles = set()

with open("event_data_word/all_tuples.txt", 'r') as f:
    file = f.readlines()

    for line in file:
        tups = line.split('|')
        for tup in tups:
            phrases = tup.split(';')
            Event_types.add(phrases[1].strip())
            Arguments.add(phrases[3].strip())
            Roles.add(phrases[4].strip())

with open("event_data_word/event_types.txt", 'w') as f:
    Event_types = list(Event_types)
    Event_types.sort()
    for L in Event_types:
        f.write(L)
        f.write("\n")

with open("event_data_word/arguments.txt", 'w') as f:
    Arguments = list(Arguments)
    Arguments.sort()
    for L in Arguments:
        f.write(L)
        f.write("\n")

with open("event_data_word/roles.txt", 'w') as f:
    Roles = list(Roles)
    Roles.sort()
    for L in Roles:
        f.write(L)
        f.write("\n")