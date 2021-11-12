# deploy ; Movement:Transport ; 17,000 U.S. Army soldiers ; PER:Group ; Artifact | deploy ; Movement:Transport ; the Persian Gulf region ; LOC:Region-International ; Destination

Event_types = set()
Arguments = set()
Roles = set()

with open("Tuples.txt", 'r') as f:
    file = f.readlines()

    for line in file:
        tups = line.split('|')
        for tup in tups:
            phrases = tup.split(';')
            Event_types.add(phrases[1].strip())
            Arguments.add(phrases[3].strip())
            Roles.add(phrases[4].strip())

print(Event_types)
print(Arguments)
print(Roles)
