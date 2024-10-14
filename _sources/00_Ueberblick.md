# Überblick
Dies ist der Beginn eines Skriptes für die Vorlesung Data Mining und Grundlagen Maschinelles Lernen I an der Hochschule Karlsruhe im Studiengang Data Science. Vorläufig sind hier nur Abschnitte über die benötigte multivariate Analysis und Optimierung zu finden. 

## Zur mathematischen Notation
Naturgemäß werden Sie in diesem Mathe-Skript mit vielen Symbolen konfrontiert. Um Ihnen den Weg durch diesen algebraischen Dschungel zu erleichtern (was ist Zahl, was ist Vektor, was ist Matrix?), wird in diesem Skript folgende Notation angewendet. 
- **Lateinische Kleinbuchstaben** wie $x$, $y$, $a$ usw. bezeichnen (meist reelle) Zahlen.
- **Fettgedruckte, lateinische Kleinbuchstaben** wie $\v x$, $\v y$, $\v a$ usw. bezeichnen Vektoren im $\R^n$. Deren Komponenten sind reelle Zahlen und werden mit einem Subscript indiziert (und sind nicht fettgedruckt, sind ja Zahlen):
  \begin{align*}
    \v x=\bmat x_1\\x_2\\\vdots\\x_n\emat
  \end{align*}
- **Fettgedruckte, lateinische Großbuchstaben** wie $\m A$, $\m X$, $\m H$ usw. bezeichnen $m\times n$-Matrizen. Deren Einträge (reelle Zahlen) sind die zugehörigen Kleinbuchstaben (nicht fettgedruckt) und werden mit zwei Subscripten indiziert:
  \begin{align*}
    \m A=\bmat a_{11} & a_{12} & \cdots & a_{1n}\\
                a_{21} & \cdots & \cdots & \\
                \vdots &  &      & \vdots\\
                a_{m1} & \cdots & \cdots & a_{mn}\emat
  \end{align*}
- **Griechische Kleinbuchstaben** wie $\alpha$, $\lambda$, $\mu$ usw. bezeichnen reelle Zahlen.
- Iterationszähler in iterativen Verfahren (Gradientenverfahren u.ä.) werden mit einem **Superscript in eckigen Klammern** indiziert: So bezeichnet $\v x^{[k]}$ den Vektor $\v x$ während der $k$-ten Iteration.


## Inhaltsverzeichnis
```{tableofcontents}
```

<!-- ## Notation
Naturgemäß werden Sie in diesem Mathe-Skript mit vielen Symbolen konfrontiert. Um Ihnen den Weg durch diesen algebraischen Dschungel zu erleichtern (was ist Zahl, was ist Vektor?), wird in diesem Skript folgende Notation angewendet. 
- **Lateinische Kleinbuchstaben** wie $x$, $y$, $a$ usw. bezeichnen (meist reelle) Zahlen.
- **Fettgedruckte, lateinische Kleinbuchstaben** wie $\v x$, $\v y$, $\v a$ usw. bezeichnen Vektoren im $\R^n$. Deren Komponenten sind reelle Zahlen und werden mit einem Subscript indiziert (und sind nicht fettgedruckt, sind ja Zahlen):
  \begin{align*}
    \v x=\bmat x_1\\x_2\\\vdots\\x_n\emat
  \end{align*}
- **Fettgedruckte, lateinische Großbuchstaben** wie $\m A$, $\m X$, $\m H$ usw. bezeichnen $m\times n$-Matrizen. Deren Einträge (reelle Zahlen) sind die zugehörigen Kleinbuchstaben (nicht fettgedruckt) und werden mit zwei Subscripten indiziert:
  \begin{align*}
    \m A=\bmat a_{11} & a_{12} & \cdots & a_{1n}\\
                a_{21} & \cdots & \cdots & \\
                \vdots &  &      & \vdots\\
                a_{m1} & \cdots & \cdots & a_{mn}\emat
  \end{align*}
- **Griechische Kleinbuchstaben** wie $\alpha$, $\lambda$, $\mu$ usw. bezeichnen reelle Zahlen.
- Iterationszähler in iterativen Verfahren (Gradientenverfahren u.ä.) werden mit einem **Superscript in eckigen Klammern** indiziert: So bezeichnet $\v x^{[k]}$ den Vektor $\v x$ während der $k$-ten Iteration.
 -->
 
## Kontakt
Wenn Sie Fehler finden oder sonstige Kommentare und Anregungen haben, schreiben Sie mir gerne eine E-Mail: dennis.janka@h-ka.de.