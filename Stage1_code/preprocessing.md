### Preprocessing

1. Merge all .tex files. Remove all comments (% ...).
	
2. \ref{fig:xxx} -> [Ref id="fig: xxx"].

3. Tables
- Add [Table] at the beginning.
- Keep header row -> [TableHeader] ...
- Convert \caption{...} -> [Caption] ...
- Convert \label{tab:xxx} -> [Label id="tab:xxx"]
- Remove table body content.
	
4. Images
- Convert \includegraphics{figure/xxx.pdf} -> [Graphic src="figure/xxx.pdf"]
- Convert \caption{...} -> [Caption] ...
- Convert \label{fig:xxx} -> [Label id="fig:xxx"]



### model selection
