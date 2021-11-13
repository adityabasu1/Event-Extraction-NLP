# deploy ; Movement:Transport ; 17,000 U.S. Army soldiers ; PER:Group ; Artifact | deploy ; Movement:Transport ; the Persian Gulf region ; LOC:Region-International ; Destination

Event_types = set()
Arguments = set()
Roles = set()

with open("all_tuples.txt", 'r') as f:
    file = f.readlines()

    for line in file:
        tups = line.split('|')
        for tup in tups:
            phrases = tup.split(';')
            Event_types.add(phrases[1].strip())
            Arguments.add(phrases[3].strip())
            Roles.add(phrases[4].strip())

with open("event_types.txt", 'w') as f:
    for L in Event_types:
        f.write(L)
        f.write("\n")

with open("arguments.txt", 'w') as f:
    for L in Arguments:
        f.write(L)
        f.write("\n")

with open("roles.txt", 'w') as f:
    for L in Roles:
        f.write(L)
        f.write("\n")