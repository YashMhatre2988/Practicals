goal([1,2,3,4,5,6,7,8,0]).

solve(S) :- hill_climb(S,[]).

hill_climb(S,_) :- goal(S), !, write('Goal Reached:'), nl, print_state(S).
hill_climb(S,V) :-
    findall(C, move(S,C), Cs),
    exclude(member_(V), Cs, Valid),
    pairs(Valid, Scored),
    sort(Scored, [[_,Best]|_]),
    write('Current:'), nl, print_state(S),
    hill_climb(Best,[S|V]).
hill_climb(S,_) :- write('Stuck:'), nl, print_state(S).

member_(L,E) :- member(E,L).

move(S,N) :- swap(S,0,-3,N).
move(S,N) :- swap(S,0, 3,N).
move(S,N) :- swap(S,0,-1,N).
move(S,N) :- swap(S,0, 1,N).

swap(S,Z,D,N) :-
    nth0(I,S,Z), J is I+D,
    J>=0, J<9, \+invalid_edge(I,J),
    nth0(J,S,V),
    set(S,I,V,T), set(T,J,Z,N).

invalid_edge(2,3). invalid_edge(3,2).
invalid_edge(5,6). invalid_edge(6,5).

set([_|T],0,X,[X|T]).
set([H|T],I,X,[H|R]) :- I>0, I1 is I-1, set(T,I1,X,R).

heuristic(S,H) :- goal(G), count_mis(S,G,H).
count_mis([],[],0).
count_mis([0|T1],[_|T2],H)  :- count_mis(T1,T2,H).
count_mis([X|T1],[X|T2],H)  :- count_mis(T1,T2,H).
count_mis([_|T1],[_|T2],H)  :- count_mis(T1,T2,H1), H is H1+1.

pairs([],[]).
pairs([S|R],[[H,S]|PR]) :- heuristic(S,H), pairs(R,PR).

print_state([A,B,C,D,E,F,G,H,I]) :-
    format("~w ~w ~w~n~w ~w ~w~n~w ~w ~w~n",[A,B,C,D,E,F,G,H,I]).