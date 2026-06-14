goal(state(2,_)).

move(state(X,Y), state(4,Y)) :- X<4.
move(state(X,Y), state(X,3)) :- Y<3.
move(state(X,Y), state(0,Y)) :- X>0.
move(state(X,Y), state(X,0)) :- Y>0.
move(state(X,Y), state(X1,Y1)) :- X>0, Y<3, T is min(X,3-Y), X1 is X-T, Y1 is Y+T.
move(state(X,Y), state(X1,Y1)) :- Y>0, X<4, T is min(Y,4-X), Y1 is Y-T, X1 is X+T.

dfs(S,_,[S]) :- goal(S).
dfs(S,V,[S|P]) :- move(S,N), \+member(N,V), dfs(N,[N|V],P).

solve(P) :- dfs(state(0,0),[state(0,0)],P).