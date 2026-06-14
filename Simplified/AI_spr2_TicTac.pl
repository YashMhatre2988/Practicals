:- set_prolog_flag(answer_write_options,[quoted(true),portray(true),max_depth(0)]).
initial([e,e,e,e,e,e,e,e,e]).

win([P,P,P,_,_,_,_,_,_],P). win([_,_,_,P,P,P,_,_,_],P). win([_,_,_,_,_,_,P,P,P],P).
win([P,_,_,P,_,_,P,_,_],P). win([_,P,_,_,P,_,_,P,_],P). win([_,_,P,_,_,P,_,_,P],P).
win([P,_,_,_,P,_,_,_,P],P). win([_,_,P,_,P,_,P,_,_],P).

goal(B) :- win(B,x).

next_player(x,o). next_player(o,x).

replace([_|T],0,X,[X|T]).
replace([H|T],I,X,[H|R]) :- I>0, I1 is I-1, replace(T,I1,X,R).

move(B,P,N) :- nth0(I,B,e), replace(B,I,P,N).

bfs([[B|Path]|_],_,[B|Path]) :- goal(B), !.
bfs([[B|Path]|Q],P,Sol) :-
    findall([N,B|Path], move(B,P,N), Kids),
    append(Q,Kids,Q2),
    next_player(P,P2),
    bfs(Q2,P2,Sol).

solve(Path) :- initial(B), bfs([[B]],x,R), reverse(R,Path).

print_board([A,B,C,D,E,F,G,H,I]) :-
    format(" ~w|~w|~w~n---+---+---~n ~w|~w|~w~n---+---+---~n ~w|~w|~w~n~n",[A,B,C,D,E,F,G,H,I]).

print_solution([]).
print_solution([B|R]) :- print_board(B), print_solution(R).