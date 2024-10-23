# lambda nfa to dfa
- convert lambda Nondeterministic finite automaton to deterministic finite automaton
- check if input string is accepted in automata
# input format
- first line is 5 number wich is: amount of input automaton states, amount of alphabet, amount of accept states, amount of transitions,amount of input string that would check is accepted in automaton
- next lines are language alphabets
- next line is initial state
- next lines are accepting states
- next lines are transitions in this format(first-state transition-symbol second-state)
- next lines are input string that would check
```
5 2 1 6 2
a
b
0
4
0 a 1
1 a 0
0 b 4
0 a 2
2 a 3
3 b 0
aaa
aab
```

# output
- converted dfa graph plot
![alt text](image.png)
- the result of input string check in the automaton
- converted dfa graph print

# input for automaton with lambda transition
```
3 3 1 5 7
a
b
c
0
2
0 a 0
0 $ 1
1 b 1
1 $ 2
2 c 2
aaa
aabbb
ccaaabbb
ccc
bbbcc
acb
ccba
```
