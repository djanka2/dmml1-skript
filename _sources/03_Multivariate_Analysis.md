---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Grundlagen Analysis

In diesem Kapitel führen wir den Begriff der *multivariaten Funktion* und der *partiellen Ableitung* ein. Diese beiden Begriffe sind von zentraler Bedeutung für die gesamte Vorlesung.

(sec:multivariate_Funktionen)=
## Multivariate Funktionen
Ein zentraler Begriff der nichtlinearen Optimierung und des maschinelles Lernen ist der Begriff der *multivariaten Funktion*. Dies ist eine Verallgemeinerung des Funktionenbegriffs, wie er aus der Analysis einer Veränderlicher (1. Semester) bekannt ist: Wie im eindimensionalen beschreibt eine multivariate Funktion eine Beziehung zwischen einer Eingabegröße und einer Ausgabegröße (dem *Funktionswert*), nur dass die Eingabegröße nicht wie im eindimensionalen Fall eine Zahl ist, sondern ein Vektor im $\R^n$, wobei $n\in\N$ im folgenden eine beliebige natürliche Zahl sei. Die Ausgabegröße kann ebenfalls ein Vektor sein, der von anderer Dimension wie der Eingangsvektor sein kann. Anschaulich:
```{figure} ./bilder/multivariate_funktion.png
:name: Multivariate Funktion $\R^n\rightarrow\R^m$ (schematisch).
:width: 400px
```

Um eine Funktion mathematisch korrekt zu spezifizieren, schreiben wir
\begin{align}
	\v f:D\subseteq\R^n\rightarrow\R^m\label{eq:func}\\
	\v x\mapsto \v f(\v x)\label{eq:mapsto}
\end{align}
Die erste Zeile bedeutet, dass $f$ eine Abbildung von einer Teilmenge $D$ des $\R^n$, der sog. *Definitionsbereich* oder die *Definitionsmenge*, nach $\R^m$ ist. $\R^m$ nennen wir in diesem Zusammenhang auch den *Bildraum*, *Zielraum* oder die *Zielmenge*. Die zweite Zeile spezifiziert, dass ein Eingabevektor $\v x\in\R^n$ auf einen Funktionsvektor $\v f(\v x)\in\R^m $ abgebildet wird. Eine multivariate Funktion $\v f$ weist jedem Eingabevektor *genau einen* Funktionsvektor $\v f(\v x)$ zu.

```{note}
Obwohl $\v x\in\R^n$ ein Spaltenvektor ist, schreibt man normalerweise $\v f(\v x)=f(x_1,x_2,\dots,x_n)$. Beim Spezialfall $f:\R^2\rightarrow\R$ nennt man die beiden Komponenten des Eingabevektors oft statt $x_1, x_2$ auch $x$, $y$ und schreibt $f(x,y)$ für den Funktionswert.
```

In der Optimierung interessiert man sich vor allem für den Fall, dass der Funktionswert eine reelle Zahl ist (also der Spezialfall $m=1$), oder in der Sprache des vorherigen Bildes:
```{figure} ./bilder/multivariate_funktion_1.png
:name: Multivariate Funktion $\R^n\rightarrow\R$ (schematisch).
:width: 400px
```
Solch eine Funktion spezifiziert man als
\begin{align}
	f:D\subseteq\R^n\rightarrow\R\\
	\v x\mapsto f(\v x)
\end{align}
Diese Funktion $f$ weist also jedem Eingabevektor *genau einen* Funktions*wert* $f(\v x)$ zu. 
%```{note}
%Je nach Kontext können also $x$ und $f(x)$ Vektoren oder Zahlen sein. 
%```

Wir betrachten zunächst einige Beispiele für multivariate Funktionen $\R^n\rightarrow\R$ bevor wir uns den grundlegenden Eigenschaften von Funktionen, nämlich Stetigkeit und Differenzierbarkeit, zuwenden.

````{prf:example} Lineare Funktion
\begin{align}
	f:\R^2\rightarrow\R\\
	\bmat x\\y\emat  \mapsto x+2y
\end{align}
````
Der Funktionsgraph dieser Funktion lässt sich als Ebene im $\R^3$ darstellen. Allgemein lassen sich Funktionen $f:\R^2\rightarrow \R$ als dreidimensionale Plots visualisieren. Dafür gibt es zwei Möglichkeiten:
- Oberflächenplot: für jeden Punkt der $x$-$y$-Ebene wird der Funktionswert $f(x,y)$ auf der dritten Achse ($z$-Achse) aufgetragen und man erhält eine Oberfläche im dreidimensionalen Raum. Für das Beispiel $f(x,y)=x+2y$:
```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import numpy as np

x = np.linspace(-4,4,100)
y = np.linspace(-2,2,100)
X,Y = np.meshgrid(x,y)
z = X + 2*Y

fig = go.Figure(go.Surface(x=x,y=y,z=z, colorscale="Blues", showscale=False))
fig.update_layout( autosize=True, height=250,
                  margin=go.layout.Margin(l=0, r=0, b=0, t=0))
```
- Konturdiagramm: Es wird die $x$-$y$-Ebene dargestellt. Entlange den eingezeichneten *Höhenlinien* (Isolinien), ist der Funktionswert jeweils gleich. Zusätzlich wird der Funktionswert durch eine Farbe gekennzeichnet.Für das Beispiel $f(x,y)=x+2y$:
```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import numpy as np

x = np.linspace(-3,3,100)
y = np.linspace(-2,2,100)
X,Y = np.meshgrid(x,y)
z = X + 2*Y

fig = go.Figure(go.Contour(x=x,y=y,z=z, colorscale="Blues", contours_coloring='heatmap'))
fig.update_layout( autosize=True, height=250,
                  margin=go.layout.Margin(l=150, r=150, b=0, t=0))
```
Funktionen von mehr als zwei Variablen lassen sich nicht mehr ohne weiteres visualisieren.

Allgemein lässt sich *jede* **lineare** Funktion $f:\R^n\rightarrow \R$ in $n$ Variablen auf folgende Form bringen:
\begin{align*}
f(x_1,x_2,\dots,x_n)=b_1x_1+\cdots +b_nx_n+c=\v b^T\v x+c
\end{align*}
wobei $b_1,\dots,b_n$ sowie $c$ (fest gewählte) reelle Zahlen sind.
Der Funktionsgraph einer linearen Funktion in $n$ Veränderlichen ist eine *Hyperebene* (d.h. ein $n$-dimensionaler Untervektorraum) im $\R^{n+1}$: 
- $n=1$: Gerade im $\R^2$
- $n=2$: Ebene im $\R^3$

````{prf:example} Quadratische Funktion
\begin{align}
	f:\R^2\rightarrow\R\\
	\bmat x\\y\emat  \mapsto 3x^2+y^2
\end{align}
````
Der Graph der Funktion $f(x,y)=3x^2+y^2$ ist eine Verallgemeinerung der Parabel, auch *Paraboloid* genannt:
```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import numpy as np

x = np.linspace(-2,2,100)
y = np.linspace(-4,4,100)
X,Y = np.meshgrid(x,y)
z = 3*X**2 + Y**2

fig = go.Figure(go.Surface(x=x,y=y,z=z, colorscale="Blues", showscale=False))
fig.update_layout( autosize=True, height=250,
                  margin=go.layout.Margin(l=0, r=0, b=0, t=0))
```
Allgemein lässt sich *jede* **quadratische** Funktion $f:\R^n\rightarrow \R$ in $n$ Variablen auf folgende Form bringen:
\begin{align*}
f(x_1,x_2,\dots,x_n)=\frac{1}{2}\v x^T\v A\v x+\v b^T\v x+c,
\end{align*}
wobei $\v A\in\R^{n\times n}$ eine symmetrische Matrix ist, $\v b\in\R^n$ ein Vektor und $c\in\R$ eine reelle Zahl. Zum Beispiel lässt sich $f(x,y)=3x^2+y^2$ schreiben als
\begin{align*}
f(x,y)=\frac{1}{2}(x,y) \bmat 6 & 0\\0&2\emat  \bmat x\\y\emat   + (0,0)\bmat x\\y\emat  +0.
\end{align*}
Wir werden uns später noch eingehender mit quadratischen Funktionen beschäftigen (dann wird auch klar werden, warum man diese zunächst kompliziert anmutende Schreibweise bevorzugt).

Natürlich kann man auch aus dem eindimensionalen bekannte Funktionen wie Polynome, Winkelfunktionen, Exponentialfunktion, Logarithmus, etc. zu mehrdimensionalen Funktionen kombinieren. Der Definitionsbereich muss eingeschränkt werden, falls Funktionen wie $\sqrt{\ }$ oder $\log$ verwendet werden.
````{prf:example} 
\begin{align}
	f:\R_{>0}\times \R_{\geq 0}\rightarrow\R\\
	\bmat x\\y\emat  \mapsto \frac{1}{2}\sin(3xy) + \log(x) - \sqrt{y}\\
\end{align}
````
Und hier der Graph dieser Funktion:
```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import numpy as np

x = np.linspace(0.1,3,100)
y = np.linspace(0.1,3,100)
X,Y = np.meshgrid(x,y)
z = 0.5*np.sin(3*X*Y) + np.log(X) - np.sqrt(Y)

fig = go.Figure(go.Surface(x=x,y=y,z=z, colorscale="Blues", showscale=False))
fig.update_layout(height=250, autosize=True,
                  margin=go.layout.Margin(l=0, r=0, b=0, t=0))
```

Schließlich möchte ich noch darauf hinweisen, dass der Begriff der *multivariaten Funktion* eine unheimlich breite und nützliche Klasse von Objekten beschreibt. So kann man z.B. *jedes* Modell des überwachten Lernens als mathematische Funktion $\v f:\R^n\rightarrow \R^m$ auffassen. So ist z.B. [DALL$\cdot$E 2](https://openai.com/dall-e-2/) eine mathematische Funktion, die eine (numerische Darstellung einer) Zeichenkette (also einen Vektor) abbildet auf den Raum der Bilder mit 1024$\times$ 1024 Pixeln (den kann man z.B. numerisch als $\R^{1024\cdot 1024\cdot 3}$ auffassen, wobei jeder Pixel durch einen dreidimensionalen RGB-Farbwert dargestellt wird). Auch DALL$\cdot$E 2 ist eine Hintereinanderausführung von---ziemlich vielen---elementaren Operationen wir Additionen, Multiplikationen und Verzweigungen.

Wir möchten als nächstes die Eigenschaften *Stetigkeit* und *Differenzierbarkeit* für multivariate Funktionen definieren. Dazu wiederholen wir kurz diese Begriffe für univariate Funktionen (siehe 1. Semester: Grundlagen der Analysis). 

(sec:diff1)=
## Differentialrechnung einer Veränderlicher
Im Folgenden setzen wir voraus, soweit nicht anders angegeben, dass alle Funktionen *stetig* sind. Zur Erinnerung: Anschaulich bedeutet das, dass man den Graph der Funktion zeichnen kann, ohne den Stift abzusetzen. Mathematisch ausgedrückt:
````{prf:definition} Stetigkeit für Funktionen einer Variablen
Eine Funktion $f:\R\rightarrow\R$ heißt *stetig* in $x_0$, wenn zu jedem $\varepsilon>0$ ein $\delta>0$ existiert, so dass für alle $x\in\R$ mit
\begin{align*}
		|x-x_0|<\delta
\end{align*}
gilt:
\begin{align*}
    |f(x)-f(x_0)|<\varepsilon
\end{align*}
````
Intuitiv bedeutet das, dass hinreichend kleine Änderungen des Arguments $x$ (nämlich kleiner als $\delta$) nur beliebig kleine Änderungen des Funktionswerts $f(x)$ verursachen.


Wir wiederholen kurz die Differentialrechnung einer Veränderlicher, die aus der Analysis Vorlesung bekannt ist (bzw. sein sollte). 
````{prf:definition} Differenzenquotient
Der *Differenzenquotient*
\begin{align}	\frac{\delta y}{\delta x}:=\frac{f(x+\delta x)-f(x)}{\delta x}	\end{align}
beschreibt die Steigung der Sekante durch zwei Punkte mit Koordinaten $x$ und $x+\delta x$.
````
Wenn wir nun $\delta x$ gegen $0$ laufen lassen (d.h. wir betrachten eine Folge von $\delta$'s, die gegen $0$ konvergiert), erhalten wir als Grenzwert die Steigung der Tangente an $f$ im Punkt $x$, falls $f$ differenzierbar ist. Die Tangente ist die Ableitung von $f$ an der Stelle $x$.

````{prf:definition} Ableitung
Die *Ableitung* von $f$ an der Stelle $x$ ist definiert als der Grenzwert
```{math}
:label: eq:ableitung
\begin{align}\label{eq:ableitung}
    \frac{\textup{d}f}{\textup{d}x}:=\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}.
\end{align}
```
````

Wenn dieser Grenzwert für einen Punkt $x$ existiert, nennen wir die Funktion *differenzierbar*. Ein Beispiel für eine Funktion, die nicht überall differenzierbar ist, ist die Betragsfunktion $f(x)=|x|$. Im Punkt $x=0$ hängt der Grenzwert {eq}`eq:ableitung` davon ab, ob man sich dem Punkt $x=0$ von links oder von rechts nähert (entweder $1$ oder $-1$). Die Ableitung der Betragsfunktion im Punkt $x=0$ ist also nicht definiert.

Für die elementaren Funktionen können wir auf gegebene Ausdrücke für die Ableitungen zurückgreifen, siehe Tabelle {numref}`tab:ableitung1`. 

```{table} Ableitungen elementarer Funktionen.
:name: tab:ableitung1

| Name                | Funktion                    | Ableitung                     |
|---------------------|-----------------------------|-------------------------------|
|Konstante            | $f(x) = c$                  | $f'(x) = 0$                   |
|Potenzen             | $f(x) = x^r$ mit $r \neq 0$ | $f'(x) = rx^{r-1}$            |
|Exponentialfunktion  | $f(x) = \exp(x)$            | $f'(x)=\exp(x)$               |
|Logarithmus          | $f(x) = \ln(x)$             | $f'(x) = \frac{1}{x} $        |
|Sinus                | $f(x) = \sin(x)$            | $f'(x) = \cos(x) $            |
|Cosinus              | $f(x) = \cos(x)$            | $f'(x) = -\sin(x) $           |
|Tangens              | $f(x) = \tan(x)$            | $f'(x) = \frac{1}{\cos^2(x)}$ |
```

Des weiteren kennen wir folgende {ref}`tab:ableitung2`, nach denen wir Ableitungen für differenzierbare Funktionen bilden dürfen.

```{table} Regeln für die Bildung von Ableitungen.
:name: tab:ableitung2

| Name | Funktion | Ableitung |
|------|----------|-----------|
|Summenregel          | $f(x) = g(x) + h(x)$ | $f'(x) = g'(x) + h'(x)$       |
|Produktregel      | $f(x) = g(x) h(x)$ | $f'(x) = g(x)h'(x) + g'(x)h(x)$ |
|Quotientenregel     | $f(x) = \frac{g(x)}{h(x)}$ | $f'(x) = \frac{g'(x)h(x) - g(x)h'(x)}{h^2(x)}$ |
|Kettenregel   | $f(x) = g(h(x))$ | $f'(x) = g'(h(x))\,h'(x)$  |
```

````{prf:example}
Wir möchten die Funktion $h(x)=(1-2x)^2$ mit der Kettenregel ableiten. Wir setzen
\begin{align*}
    &h(x)=(1-2x)^2=g(f(x))\\
    &f(x)=1-2x\\
    &g(f)=f^2.
\end{align*}
Nach den Regeln in Tabelle {numref}`tab:ableitung1` leiten wir ab:
\begin{align*}
    &f'(x)=-2\\
    &g'(f)=2f.
\end{align*}
Nach der Kettenregel ist die Ableitung von $h$ gegeben als
\begin{align*}
    h'(x)=g'(f)\cdot f'(x)=2f \cdot (-2) = 2(1-2x) \cdot (-2) = 8x-4.
\end{align*}
````
Die Kettenregel für zusammengesetzte Funktionen lässt sich auch kurz zusammenfassen als "äußere Ableitung $\times$ innere Ableitung".


## Normen und Abstände von Vektoren
Wir möchten nun Funktionen betrachten, die nicht nur von einer skalaren Variablen $x\in\R$ abhängen, sondern von mehr als einer Variablen, also von einem Vektor $\v x=(x_1,x_2,\dots,x_n)^T\in\R^n$. 

Ein entscheidender Unterschied zwischen Vektoren und Zahlen ist, dass man zwei Zahlen mittels der $<$ oder $>$ miteinander *vergleichen* kann; sie bilden eine geordnete Menge. Bei Vektoren mit $n\geq 2$ geht das nicht, Aussagen wie $\v x>\v y$ machen keinen Sinn. Man kann aber ihren *Abstand* miteinander vergleichen. Im Falle zweier Zahlen $a$ und $b$ ist ihr Abstand durch ihre Entfernung auf dem Zahlenstrahl gegeben, d.h. durch den Betrag ihrer  Differenz, $|a-b|$. Diesen Abstandsbegriff verallgemeinert man auf Vektoren mit dem Konzept der *Norm*.

% Anmerkung: wenn man \norm{x} schreibt, werden die Striche || || sehr klein dargestellt. Workaroung \norm{x_{}} ergänz ein leeres Subskript, was dazu führt, dass die Striche verlängert werden. 
````{prf:definition} Norm
Eine *Norm* auf einem Vektorraum $V$ ist eine Funktion
\begin{align*}
    \norm{\cdot_{}}: V&\rightarrow \R\\
    \v x&\mapsto \norm{\v x_{}},
\end{align*}
die jedem Vektor $\v x$ seine *Länge* $\norm{\v x_{}}\in\R$ zuweist, so dass für alle $\lambda\in\R$ und $\v x,\v y\in V$ folgendes gilt:
- Homogenität: $\norm{\lambda \v x}=|\lambda|\norm{\v x_{}}$
- Dreiecksungleichung: $\norm{\v x+\v y} \leq \norm{\v x_{}}+\norm{\v y}$
- Positive Definitheit: $\norm{\v x_{}}\geq 0$ und $\norm{\v x_{}}=0$ dann und nur dann wenn $\v x=0$
````
Der Begriff der Norm ist sehr allgemein. Für unsere Zwecke sind vor allem zwei Beispiele wichtig:
````{prf:example} Manhattan Norm
Die *Manhattan Norm* von $\v x\in\R^n$ ist definiert als
\begin{align*}
\norm{\v x_{}}_1 := \sum_{i=1}^n |x_i|,
\end{align*}
wobei $| \cdot|$ den Betrag bezeichnet.
````

````{prf:example} Euklidische Norm
Die *Euklidische Norm* von $\v x\in\R^n$ ist definiert als
\begin{align*}
\norm{\v x}_2 := \sqrt{\sum_{i=1}^n x_i^2 = \sqrt{\v x^T \v x}}
\end{align*}
und berechnet die *Euklidische Distanz* von $x$ vom Koordinatenursprung. Die Euklidische Norm nennt man auch $\ell_2$-Norm. 
````
Das folgende Bild zeigt alle Vektoren (Punkte) mit $\norm{\v x_{}}_1=1$ (rot) und alle Vektoren (Punkte) mit $\norm{\v x_{}}_2=1$ (blau):

```{figure} ./bilder/l1_l2_ball.png
:width: 400px
```

Mit der Definition der Euklidischen Norm können wir nun auch *Distanzen* (oder *Abstände*) zwischen Vektoren sinnvoll definieren:
````{prf:definition} Abstand
Für zwei Vektoren $\v x,\v y\in\R^n$ nennen wir 
\begin{align*}
d(\v x,\v y)=\norm{\v x-\v y}
\end{align*}
die *Distanz* oder den *Abstand* von $x$ und $y$. Man nennt
- $\norm{\v x-\v y}_1$ die *Manhattan-Distanz*
- $\norm{\v x-\v y}_2$ die *Euklidische Distanz*
````
Das folgende Bild veranschaulicht die beiden Distanzbegriffe: Die euklidische Distanz (blau) ist die Länge der Strecke zwischen den beiden schwarzen Punkten---genau so, wie man Abstände in der klassischen, durch Euklid[^fn:Euklid] verbreiteten Geometrie (die aus dem Schulunterricht...) misst. Im Beispiel: $\norm{\v x-\v y}_2=\sqrt{(5-1)^2+(4-1)^2}=3$. Die Manhattan-Distanz ist die Summe der beiden roten Strecken---also die Strecke, die ein Taxi zwischen den Häuserblocks Manhattans zurücklegen müsste, um vom einen zum anderen Punkt zu gelangen. Im Beispiel: $\norm{\v x-\v y}_1=|5-1|+|4-1|=7$.
[^fn:Euklid]: Griechischer Mathematiker, der wahrscheinlich im 3. Jahrhundert v. Chr. in Alexandria gelebt hat. Sein Werk *Elemente* fasst Arithmetik und Geometrie seiner Zeit zusammen. Die *Elemente* wurden 2000 Jahre lang als akademisches Lehrbuch benutzt und waren bis in die zweite Hälfte des 19. Jahrhunderts das nach der Bibel meistverbreitete Werk der Weltliteratur.
```{figure} ./bilder/l1_l2_distance.png
:width: 400px
```

## Stetigkeit multivariater Funktionen
Mit diesem Abstandsbegriff ausgerüstet, definieren wir nun den Begriff der Stetigkeit, analog zum eindimensionalen Fall. Wir tun dies hier direkt für den allgemeinsten Fall, d.h. Abbildungen von Vektoren auf Vektoren.

````{prf:definition} Stetigkeit für Funktionen mehrerer Variablen
Eine Funktion $\v f:\R^n\rightarrow\R^m$ heißt *stetig* in $\v x_0$, wenn zu jedem $\varepsilon>0$ ein $\delta>0$ existiert, so dass für alle $x\in\R^n$ mit
\begin{align*}
    \left\|\v x-\v x_0\right\|<\delta
\end{align*}
gilt:
\begin{align*}
    \left\|\v f(\v x)-\v f(\v x_0)\right\|<\varepsilon.
\end{align*}
````

Wir benutzen hier nun nicht mehr den Betrag als Abstand zwischen zwei Zahlen, sondern eine Vektornorm (z.B. die euklidische Norm), mit der man den Abstand zwischen zwei Vektoren bestimmen kann (bzw. die Länge des Differenzenvektors). Anschaulich bedeutet dies genau das gleiche wie im Eindimensionalen: wenn zwei Vektoren nahe beieinander sind, dann sind auch ihre Funktionswerte nahe beieinander (und zwar *beliebig* nahe, nämlich $<\varepsilon$, wenn nur die Vektoren nahe genug beieinander gewählt, nämlich mit Abstand $<\delta$). Oder auch: Die Funktion macht keine Sprünge.

```{prf:example} Beispiele stetiger Funktionen
Die folgenden Funktionen sind stetig:
- Lineare Funktionen sind stetig.
- Quadratische Funktionen sind stetig.
- Allgemein: Multivariate Polynome beliebigen Grades sind stetig:
\begin{align*}
p(x_1,\dots,x_n)=\sum_{q_1+q_2+\cdots+q_n\leq n}\alpha_{q_1,\dots,q_n}x_1^{q_1}\cdots x_n^{q_n}
\end{align*}
also z.B.
\begin{align*}
p(x_1,x_2,x_3)=x_1^2+x_2^4x_3+x_3^3+x_2x_1+3
\end{align*}
- Projektion auf die $j$-te Komponente $f(\v x)=x_j$ ist stetig.
- $\sin x, \cos x, \exp x$ sind stetig auf $\R$, $\ln x$ ist stetig auf $\R_{>0}$.
```

```{prf:example} Beispiele unstetiger Funktionen (I)
Die *Heaviside*-Funktion
\begin{align*}
f(x)=\left\{\begin{array}{lr} 0, & x<0\\1, & x\geq 0\end{array}\right.
\end{align*}
ist unstetig in $0$.
```
```{code-cell} ipython3
:tags: [hide-input]
import plotly.express as px
import numpy as np

x = np.array([-1,-1e-16,1e-16,1])
y = (x>=0).astype("float")
px.line(x=x,y=y, color_discrete_sequence=list(reversed(px.colors.sequential.Blues)), height=250)
```

```{prf:example} Beispiele unstetiger Funktionen (II)
Die Funktion 
\begin{align*}
f(x)=\left\{\begin{array}{lr} \sin(1/x), & x\neq 0\\0, & x=0\end{array}\right.
\end{align*}
ist unstetig in $0$.
```
```{code-cell} ipython3
:tags: [hide-input]
import plotly.express as px
import numpy as np

x = np.linspace(-0.1,0.1,10000)
y = np.sin(1/x)
px.line(x=x,y=y, color_discrete_sequence=list(reversed(px.colors.sequential.Blues)), height=250)
```

```{prf:example} Beispiele unstetiger Funktionen (III)
:label: ex:unstetig

Die Funktion 
\begin{align*}
f(x,y)=\left\{\begin{array}{lr} 0, & x=y=0\\\frac{xy}{x^2+y^2}, & \text{sonst}\end{array}\right.
\end{align*}
ist unstetig in $(0,0)$ (obwohl sie stetig in $x$ und in $y$ ist!). Je nachdem, aus welcher Richtung man sich dem Punkt $(0,0)$ nähert, ergeben sich unterschiedliche stetige Fortsetzungen der Funktion. Nähert man sich entlang einer der Koordinatenachsen, ergibt sich 0. Nähert man sich entlang der Winkelhalbierenden der $x$-$y$-Ebene, erhält man entweder $+0.5$ oder $-0.5$. 
```
```{code-cell} ipython3
:tags: [hide-input]
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
X,Y = np.meshgrid(x,y)
z = X*Y/(X**2 + Y**2)

fig = go.Figure(go.Surface(x=x,y=y,z=z, colorscale="Blues", showscale=False))
fig.update_layout( autosize=True, height=250,
                  margin=go.layout.Margin(l=0, r=0, b=0, t=0))
```

```{prf:example} Beispiele unstetiger Funktionen (IV)
Ein pathologisches Beispiel für eine unstetige Funktion ist die Dirichlet-Funktion[^fn:Dirichlet]: 
\begin{align*}
D:\R\rightarrow \R,\quad
x\mapsto D(x)=\left\{\begin{array}{ll} 1, & \text{wenn }x\text{ rational,}\\
0, &  \text{wenn }x\text{ irrational.}\end{array}\right.
\end{align*}
Diese ist NIRGENDS STETIG 🙀!
```

In vielen Fällen weiß man, dass eine Funktion stetig ist, weil sie aus stetigen Funktion "aufgebaut" ist. Das stellt der folgende Satz sicher.
````{prf:theorem}
Sind die Funktionen $f,g:D\subseteq \R^n\rightarrow \R$ stetig an der Stelle $\v x_0\in D$, dann sind auch $f+g$, $f\cdot g$ und $\frac{f}{g}$ (falls $g(\v x_0)\neq 0$) stetig.

Ist $\v f:D_f\subseteq \R^n\rightarrow \R^m$ stetig an der Stelle $\v x_0\in D_g$ und $\v g:D_g\subseteq \R^m\rightarrow \R^k$ stetig in $\v f(\v x_0)$, so ist auch die Verkettung $\v g\circ \v f$ (erst $\v f$, dann $\v g$) stetig.
````
[^fn:Dirichlet]: https://de.wikipedia.org/wiki/Dirichlet-Funktion

%😵


## Partielle Ableitungen und Gradient
Geometrisch entspricht die Ableitung einer univariaten Funktion in einem Punkt $x_0$ der  *Steigung der Tangenten* in diesem Punkt. Der Punkt $x_0$ und die Ableitung $f'(x_0)$ definieren also die Tangente.
```{figure} ./bilder/tangente.png
:width: 300px
```

Diese Anschauung lässt sich nun auch auf zwei Dimensionen übertragen: Im Zweidimensionalen definieren der Punkt $\v x_0$ und die (eindimensionalen) Ableitungen in Richtung der beiden Koordinatenachsen die Tangentialebene an die Funktion.
```{figure} ./bilder/tangentialebene.png
:width: 400px
```

Die Ableitungen in Richtung der Koordinatenachsen nennt man *partielle Ableitungen* der Funktion. Man bestimmt sie, indem *jeweils eine* der Variablen variiert wird, während die anderen konstant bleiben. Alle so bestimmten partiellen Ableitungen werden in einem Vektor angeordnet, den man *Gradient* nennt.

````{prf:definition} Partielle Ableitung und Gradient
Für eine Funktion $f:\R^n\rightarrow \R$, $\v x\mapsto f(\v x)$, $\v x\in\R^n$ in $n$ Variablen $x_1,x_2,\dots,x_n$ definieren wir die *partielle Ableitung* als
\begin{align*}
    \frac{\partial f}{\partial x_1}&=\lim_{h\rightarrow 0}\frac{f(x_1+h,x_2,\dots,x_n)-f(\v x)}{h}\\
    \frac{\partial f}{\partial x_2}&=\lim_{h\rightarrow 0}\frac{f(x_1,x_2+h,\dots,x_n)-f(\v x)}{h}\\
    &\vdots\\
    \frac{\partial f}{\partial x_n}&=\lim_{h\rightarrow 0}\frac{f(x_1,x_2,\dots,x_n+h)-f(\v x)}{h}
\end{align*}
und fassen sie in einem Zeilenvektor
```{math}
:label: eq:gradient
\begin{align}\label{eq:gradient}
    \nabla_{\v x} f=\frac{\textup{d}f}{\textup{d}\v x}=
    \left(\frac{\partial f(\v x)}{\partial x_1}\quad \frac{\partial f(\v x)}{\partial x_2}\quad\dots\quad\frac{\partial f(\v x)}{\partial x_n}\right)\in \R^{1\times n}
\end{align}
```
zusammen. Der Zeilenvektor {eq}`eq:gradient` heißt *Gradient* von $f$ und ist eine Verallgemeinerung der Ableitung aus Abschnitt {ref}`sec:diff1`. Das bedeutet, für $n=1$ ist der Gradient die gewöhnliche Ableitung in einer Variablen. Wenn klar ist, nach welchen Variablen abgeleitet wird, schreiben wir anstatt $\nabla_{\v x} f$ auch einfach $\nabla f$.
````
 

Wenn man sich Formel {eq}`eq:gradient` genauer anschaut, sieht man, dass jede partielle Ableitung eine Ableitung nach einem Skalar ist---also genau der Ableitungsbegriff, den wir schon aus dem eindimensionalen kennen. Anschaulich ist die partielle Ableitung $\derv{f}{x_1}$ die Änderung (genauer: die Änderungs*rate*) des Funktionswertes bei kleinen Störungen von $x_1$, während die anderen Variablen konstant gehalten werden.

````{prf:example}
Für $f(x_1,x_2)=x_1^2+x_2^2$ erhalten wir die partiellen Ableitungen
\begin{align*}
    \frac{\partial f}{\partial x_1} = 2x_1\\
    \frac{\partial f}{\partial x_2} = 2x_2,
\end{align*}
d.h. $\nabla f(x_1,x_2)=\left(2x_1,2x_2\right)$.
````

````{prf:example}
Für $f(x_1,x_2)=x_1^2\cdot x_2$ erhalten wir die partiellen Ableitungen
\begin{align*}
    \frac{\partial f}{\partial x_1} = 2x_1 x_2\\
    \frac{\partial f}{\partial x_2} = x_1^2,
\end{align*}
d.h. $\nabla f(x_1,x_2)=\left(2x_1 x_2, x_1^2\right)$.
````

````{prf:example}
Für $f(x_1,x_2)=(x_1+2x_2^3)^2$ erhalten wir die partiellen Ableitungen
\begin{align*}
    \frac{\partial f}{\partial x_1} = 2(x_1+2x_2^3)\frac{\partial}{\partial x_1}(x_1+2x_2^3)=2(x_1+2x_2^3)\\
    \frac{\partial f}{\partial x_2} = 2(x_1+2x_2^3)\frac{\partial}{\partial x_2}(x_1+2x_2^3)=12(x+2x_2^3)x_2^2.
\end{align*}
Für beide partiellen Ableitungen haben wir die Kettenregel für univariate Funktionen benutzt.
````

````{prf:example}
Für $f(x_1,x_2)=w_0+w_1x_1+w_2x_2$ erhalten wir die partiellen Ableitungen
\begin{align*}
    \frac{\partial f}{\partial x_1} = w_1\\
    \frac{\partial f}{\partial x_2} = w_2,
\end{align*}
d.h. $\nabla f(x_1,x_2)=(w_1,w_2)$.
````

````{prf:example}
Für $f(x_1,\dots,x_n)=w_0+\v w^T\v x=w_0+w_1x_1+\cdots + w_nx_n$ erhalten wir die partiellen Ableitungen
\begin{align*}
    \frac{\partial f}{\partial x_1} &= w_1\\
    &\vdots\\
    \frac{\partial f}{\partial x_n} &= w_n,
\end{align*}
d.h. $\nabla f(x_1,\dots, x_n)=(w_1,\dots,w_n)$.
````

### Zusammenhang zwischen Stetigkeit und Differenzierbarkeit
Für univariate Funktionen gilt folgender Zusammenhang:
\begin{align*}
\text{Stetig differenzierbar} \Rightarrow \text{differenzierbar} \Rightarrow \text{stetig}
\end{align*}
Das bedeutet, wenn eine Funktion an einem Punkt differenzierbar ist, muss sie dort auch stetig sein (aber die Ableitung muss nicht zwingend stetig sein). Oder andersherum: wenn die Funktion an einem Punkt einen Sprung macht, kann man sie dort nicht ableiten. 

Für multivariate Funktionen folgt aus der partiellen Differenzierbarkeit allerdings noch nicht, dass die Funktion stetig sein muss, sondern nur, dass sie in Richtung der Koordinatenachsen (auf die beziehen sich ja die partiellen Ableitungen) stetig ist. Stetigkeit folgt aber dann, wenn alle partiellen Ableitungen (als Funktionen aufgefasst) stetig sind:
\begin{align*}
\text{Stetig partiell differenzierbar} \Rightarrow \text{stetig}
\end{align*}
Das bedeutet, dass es auch nicht stetige Funktionen geben kann, die aber trotzdem partiell abgeleitet werden können.
````{prf:example}
:label: ex:partiell-unstetig
Die Funktion aus {prf:ref}`ex:unstetig`
\begin{align*}
f(x,y)=\left\{\begin{array}{lr} 0, & x=y=0\\\frac{xy}{x^2+y^2}, & \text{sonst}\end{array}\right.
\end{align*}
ist im Punkt $(0,0)$ partiell differenzierbar (die beiden partiellen Ableitungen $\partial_x f$ und $\partial_y f$ sind dort $0$), aber sie ist dort nicht stetig.
````

<!-- In {numref}`sec:theo` schauen wir uns noch den Begriff der *totalen Differenzierbarkeit* an, eine alternative Möglichkeit, den Begriff der Differenzierbarkeit einzuführen (und aus dem, im Gegensatz zur partiellen Differenzierbarkeit, auch die Stetigkeit folgt). -->


## Ableitungen von vektorwertigen Funktionen: Jacobi-Matrix
Bisher haben wir nur Ableitungen von reellwertigen Funktionen $f:\R^n\rightarrow\R$ betrachtet. Wir können aber genausogut Funktionen betrachten die einen Vektor auf einen Vektor abbilden: $\v f:\R^n\rightarrow\R^m$, mit $m,n\geq 1$ (beachte den Unterschied in der Schreibweise: $\v f$: vektorwertige Funktion vs. $f$: reellwertige Funktion). In diesem Fall schreiben wir den *Vektor* der Funktionswerte als
```{math}
:label: eq:fktvektor
\begin{align}
	\v f(\v x)=\bmat 
		f_1(\v x)\\ \vdots \\f_m(\v x)
	\emat  \in \R^m.
\end{align}
```
Das bedeutet, wir können die Vektor-wertige Funktion $\v f$ als Vektor von reellwertigen Funktionen $\left(f_1,\dots,f_m\right)\in\R^m$ auffassen. 
%Die Ableitungsregeln für jedes $f_i$ können wir wieder auf die Regeln {eq}`eq:sumrule`, {eq}`eq:productrule` und {eq}`eq:chain` (siehe nächster Abschnitt) zurückführen. 
Damit ist auch die Ableitung von Vektor-wertigen Funktionen sehr ähnlich zur Ableitung von eindimensionalen Funktionen; man muss lediglich aufpassen, dass man die partiellen Ableitungen richtig sortiert und den Überblick über die Indizes behält.

Analog zum Gradienten werden die partiellen Ableitungen einer Funktion $\v f:\R^n\rightarrow \R^m$ in der *Jacobimatrix* zusammengefasst, einer Verallgemeinerung des Gradientenbegriffes.
````{prf:definition} Jacobimatrix
Sei $\v f:\R^n\rightarrow \R^m$ eine differenzierbare Funktion. Dann heißt die $m\times n$ Matrix $\v J$ der partiellen Ableitungen,
\begin{align}
    \v J=\nabla \v f(\v x) = \bmat 
        \nabla f_1(\v x)\\
        \vdots\\
        \nabla f_m(\v x)
    \emat  =\bmat 
        \derv{f_1}{x_1}& \cdots & \derv{f_1}{x_n}\\
        \vdots\\
        \derv{f_m}{x_1}& \cdots & \derv{f_m}{x_n}
    \emat  \in \R^{m\times n}
\end{align}
die *Jacobimatrix* von $\v f$.
````

````{prf:example}
:label: ex:jacobi
Sei $\v f:\R^3\rightarrow \R^2$ eine Funktion gegeben durch
\begin{align*}
    \v f(x_1,x_2,x_3)=\bmat 
        x_1x_2+x_3 \\x_3^2+1 
    \emat  
\end{align*}
Dann ist die Jacobimatrix $J$:
\begin{align*}
    \v J=\bmat 
        x_2 & x_1 & 1 \\
        0  & 0  &2x_3
    \emat  
\end{align*}
````


````{prf:example}
:label: bsp:linear
Wir betrachten die Funktion $\v f:\R^m\rightarrow \R^n$ mit 
\begin{align*}
    \v f(\v x)=\v A\v x,\quad \v f(\v x)\in\R^m,\quad \v A\in\R^{m\times n},\quad \v x\in\R^n
\end{align*}
Nach der Definition der Jacobimatrix muss der Gradient dieser Funktion eine $m\times n$ Matrix sein: $\nabla \v f\in\R^{m\times n}$. Nun berechnen wir die partiellen Ableitungen von jedem der $m$ Einträge des Ergebnisvektors $\v f$ nach jeder der $n$ Variablen $x_j$. Nach der Definition des Matrix-Vektor Produkts gilt
\begin{align*}
    f_i(\v x)=\sum_{j=1}^n a_{ij}x_j \Rightarrow \derv{f_i}{x_j}= a_{ij}
\end{align*}
Wir fassen alle partiellen Ableitungen in der Jacobimatrix zusammen und erhalten:
\begin{align*}
    \nabla f = \bmat 
        \derv{f_1}{x_1}& \cdots & \derv{f_1}{x_n}\\
        \vdots\\
        \derv{f_m}{x_1}& \cdots & \derv{f_m}{x_n}
    \emat  =
    \bmat 
        a_{11} &\cdots & a_{1n}\\
        \vdots & 		& \vdots\\
        a_{m1} & \cdots & a_{mn}
    \emat  =\v A\in \R^{m\times n}
\end{align*}
Das bedeutet, dass die Ableitung einer multivariaten linearen Funktion gerade die Koeffizientenmatrix ist. Dies entspricht im Eindimensionalen der Tatsache, dass die Ableitung einer Geradengleichung (=lineare Funktion im Eindimensionalen) die Geradensteigung (=Koeffizient von $x$) is.
````

## Die Kettenregel
Für multivariate Funktionen $f:\R^n\rightarrow \R$ gelten die gleichen Regeln wie für univariate Funktionen: Summenregel, Produktregel und Kettenregel, siehe Tab. {numref}`tab:ableitung2`. Summenregel und Produktregel für zwei Funktionen $f,g:\R^n\rightarrow \R$ sind wie folgt:
- Summenregel:
```{math}
:label: eq:sumrule
    \begin{align}\label{eq:sumrule}
		&\derv{}{\v x}\left(f(\v x)+g(\v x)\right)=\derv{f}{\v x}+\derv{g}{\v x}
	\end{align}
```
- Produktregel:
```{math}
:label: eq:productrule
	\begin{align}\label{eq:productrule}
		&\derv{}{\v x}\left(f(\v x)g(\v x)\right)=\derv{f}{\v x}g(\v x)+f(\v x)\derv{g}{\v x}
	\end{align}
```
Für vektorwertige Funktionen gelten diese Regeln für jede Komponente des Vektors der Funktionswerte.

Ein wenig schwieriger ist die Kettenregel. Dort werden im univariaten Fall Ableitungen multipliziert ("äußere $\times$ innere Ableitung"). Im multivariaten Fall haben wir es aber mit Ableitungen nach *Vektoren* $\v x\in\R^n$ zu tun, d.h., in den Ableitungen tauchen nun Vektoren und Matrizen auf und deren Multiplikation ist nicht immer definiert und auch nicht kommutativ (d.h. im Allgemeinen ist für zwei Matrizen $\v A$ und $\v B$ das Produkt $\v A\v B\neq \v B\v A$).

Wir geben die Kettenregel zunächst formal an. Für zwei differenzierbare Funktionen mit passenden Definitions- und Bildbereichen, d.h. $\v f:\R^n\rightarrow \R^m$, $\v g:\R^m\rightarrow\R^k$ gilt:
```{math}
:label: eq:chain
	\begin{align}\label{eq:chain}
		&\derv{}{\v x}\left(\v g\circ \v f\right)(\v x)=\derv{}{\v x}\left(\v g(\v f(\v x))\right)=\derv{\v g}{\v f(\v x)}\derv{\v f}{\v x}
	\end{align}
```
Als Denkhilfe kann man sich die Kettenregel wie folgt gut merken: In Gleichung {eq}`eq:chain` ist nach der Ableitung von $\v g$ nach $\v x$ gesucht. Im Term der rechten Seite "kürzt" sich das $\partial \v f$ in dem "Bruch" $\derv{\v g}{\v f}\derv{\v f}{\v x}$ gerade weg, so dass am Ende $\derv{\v g}{\v x}$, also die gesuchte Größe, "übrig bleibt". Das Ganze ist aber wirklich nur eine Denkstütze, denn: $\derv{\v f}{\v x}$ ist *kein* Bruch, sondern lediglich eine Schreibweise (für die partielle Ableitung). 

Im folgenden möchten wir die Kettenregel zunächst an einem (wichtigen) Spezialfall näher betrachten. Nehmen wir an, $g:\R^2\rightarrow\R$ sei eine Funktion in zwei Variablen $f_1, f_2$. Weiterhin seien $f_1(t)$ und $f_2(t)$ selbst Funktionen von $t\in\R$. Wir wenden nun die Kettenregel an, um den Gradienten von $g$ nach $t$ zu berechnen. Das ist eine Abbildung von $\R$ nach $\R$: Ein Skalar $t$ wird zunächst abgebildet auf einen Vektor $\v f(t)=(f_1(t), f_2(t))^T$. Dieser Vektor wird dann abgebildet auf einen skalaren Funktionswert $g(f_1(t),f_2(t))$. Anschaulich bedeutet dies: wie ändert sich $g$ bei kleinen Änderungen von $t$.
```{math}
:label: eq:chain1
\begin{align}\label{eq:chain1}
	\nabla_t f = \bmat 
		\derv{g}{f_1(t)} & \derv{g}{f_2(t)}
	\emat  \bmat 
		\derv{f_1(t)}{t}\\\derv{f_2(t)}{t}
	\emat  =\derv{g}{f_1}\derv{f_1}{t}+\derv{g}{f_2}\derv{f_2}{t}
\end{align}
```
Gleichung {eq}`eq:chain1` ist das Produkt von einem Zeilenvektor und einem Spaltenvektor mit jeweils zwei Einträgen. Hier wird deutlich, dass es wichtig ist, auf die korrekte Reihenfolge und die Dimensionen zu achten, da sonst u.U. das Produkt gar nicht definiert ist. Auch hier können wir, wie im Eindimensionalen, die Kettenregel zusammenfassen als ``äußere Ableitung ($g$ nach $\v f(t)$) $\times$ innere Ableitung ($\v f$ nach $t$)''.
Wenden wir nun Gleichung {eq}`eq:chain1` in einem konkreten Beispiel an.

````{prf:example}
Sei $g\left(f_1,f_2\right)=f_1^2+2f_2$, wobei $f_1=\sin t$ und $f_2=\cos t$, dann ist
\begin{align}
    \nabla_t g &=\derv{g}{f_1}\derv{f_1}{t}+\derv{g}{f_2}\derv{f_2}{t}\\
    &=2f_1 \derv{f_1}{t} + 2\derv{f_2}{t}\\
    &=2\sin t \derv{\sin t}{t} + 2\derv{\cos t}{t}\\
    &=2\sin t \cos t - 2\sin t
\end{align}
die Ableitung von $g$ nach $t$ (in diesem Fall kann man sich $g$ als eine eindimensionale Funktion vorstellen, die ein Skalar über einen ``Umweg'' in den $\R^2$ abbildet und dann wieder zurück nach $\R$).
````

Das Verständnis und die Anwendung der Kettenregel erfordert etwas Übung. Die Kettenregel ist allerdings ein wichtiges Instrument bei der automatischen, exakten Berechnung von Ableitungen mittels algorithmischer Differentiation. Wir betrachten ein weiteres Beispiel. Viele weitere Beispiele gibt es in den Übungen.
````{prf:example}
Sei $\v f$ gegeben wie in {prf:ref}`ex:jacobi`. Zusätzlich sei $g:\R^2\rightarrow\R$ gegeben durch
\begin{align*}
    g(x_1,x_2) = x_1^2x_2
\end{align*}
Die Ableitung der Funktion $g\circ \v f$ berechnet sich wie folgt:
```{math}
:label: eq:matmult
\begin{align}
    \nabla (g(\v f(\v x))&= \nabla g(\v f(\v x))\cdot \nabla \v f(\v x)\\
    &=\nabla g(f_1(\v x),f_2(\v x))\cdot\nabla \v f(\v x)\\
    &=\bmat 
        \derv{g}{f_1(\v x)}&\derv{g}{f_2(\v x)}
    \emat  \bmat 
        x_2 & x_1 & 1 \\
        0  & 0  &2x_3
    \emat  \\
    &=\bmat 
        2f_1(x)f_2(x), &f_1(x)^2
    \emat  
    \bmat 
        x_2 & x_1 & 1 \\
        0  & 0  &2x_3
    \emat  \label{eq:matmult}\\
    %						&= \bmat 
        %							2(x_1x_2+x_3)(x_3^2+1)&(x_1x_2+x_3)^2
        %						\emat  
    %					\bmat 
        %						x_2 & x_1 & 1 \\
        %						0  & 0  &2x_3
        %					\emat  		
    &=\bmat 
        2f_1(x)f_2(x)x_2, & 2f_1(x)f_2(x)x_1, & 2f_1(x)f_2(x) + f_1(x)^2 2x_3
    \emat  \label{eq:result1}
    %		x_1x_2+x_3 \\
    %x_3^2+1 
\end{align}
```
Nun kann man in die letzte Gleichung für die Einträge $f_1(\v x)$ und $f_2(\v x)$ noch die entsprechenden Funktionswerte von $\v f$ einsetzen und erhält einen Ausdruck in $x_1$, $x_2$, $x_3$:
```{math}
:label: eq:result2
\begin{align}
    &\nabla (g(\v f(\v x))=\\
    &\bmat 
        2(x_1x_2+x_3)(x_3^2+1)x_2, & 2(x_1x_2+x_3)(x_3^2+1)x_1, & 2(x_1x_2+x_3)(x_3^2+1) + (x_1x_2+x_3)^2 2x_3
    \emat  \label{eq:result2}
\end{align}
```

Man sieht: die Funktion $g\circ \v f$ bildet einen Vektor aus dem $\R^3$ nach $\R$ ab. Nach der Matrixmultiplikation in {eq}`eq:matmult` erhält man folglich einen Zeilenvektor mit 3 Einträgen als Gradienten {eq}`eq:result2`.
````


<!-- ## Ableitungen höherer Ordnung
Auch im multivariaten Fall können wir Ableitungen höherer Ordnung betrachten. Wir tun dies hier nur für reellwertige Funktionen $f:\R^n\rightarrow\R$. Dazu fassen wir den Gradienten von $f$ auch als Funktion auf: 
```{math}
:label: eq:gradfkt
\begin{align}
	\nabla f^T: \R^n \rightarrow \R^n\\
	\v x\mapsto \bmat 
		\derv{f}{x_1} \\ \cdots \\ \derv{f}{x_n}
	\emat  \label{eq:gradfkt}
\end{align}
```
Beachte, dass wir hier den transponierten Gradienten benutzen, um einen Spaltenvektor zu erhalten, in Übereinstimmung mit dem Ausdruck {eq}`eq:fktvektor`. Wenn wir nun die partielle Ableitung dieser partiellen Ableitung bilden (also die Jacobimatrix der Gradientenfunktion) erhalten wir die zweite partielle Ableitung. Gemäß der Definition der Jacobimatrix ist das eine Matrix mit $n$ Zeilen (da {eq}`eq:gradfkt` $n$ Komponenten hat) und $n$ Spalten (da {eq}`eq:gradfkt` eine Funktion in $n$ Variablen ist).

Wir benutzen die folgenden Schreibweisen:
- $\dervquad{f}{x}$ ist die zweite partielle Ableitung von $f$ nach $x$
- $\dervzwei{f}{y}{x}=\derv{}{y}(\derv{f}{x})$ ist die partielle Ableitung von $f$, bei der erst nach $x$ und dann nach $y$ abgeleitet wird
- $\dervzwei{f}{y}{x}=\derv{}{x}(\derv{f}{y})$ ist die partielle Ableitung von $f$, bei der erst nach $y$ und dann nach $x$ abgeleitet wird
Die partiellen zweiten Ableitungen werden in der *Hessematrix* zusammengefasst. Für eine Funktion zweier Variablen $x$ und $y$ schreibt man
\begin{align}
	\nabla_{xy}^2f=H=\bmat 
		\dervquad{f}{x} & \dervzwei{f}{y}{x} \\
		\dervzwei{f}{x}{y} & \dervquad{f}{y}
	\emat  
\end{align}
Alle Aussagen dieses Abschnitts gelten auch für $n$ Variablen. Man hat dann entsprechend $n$ statt $2$ Zeilen und Spalten.
Achtung: im Allgemeinen ist die Hessematrix einer Funktion $f:\R^n\rightarrow\R$ selbst eine Funktion, und zwar $\v H:\R^n\rightarrow \R^{n\times n}$. Ächz. Das heißt für uns vor allem, dass $\v H$ keine feste Matrix ist, sondern dass sich die Einträge ändern, je nachdem, an welchem Punkt $\v x$ wir uns die Hessematrix anschauen.

Zwei wichtige Ausnahmen bilden lineare und quadratische Funktionen.
````{prf:example} Besondere Hessematrizen

Hessematrix einer linearen Funktion
: Jede lineare Funktion lässt sich schreiben als
$f(x_1,\dots,x_n)=b_1x_1+\cdots+b_nx_n+c$. Da der Gradient $\nabla f(x_1,\dots,x_n)=(b_1,\dots,b_n)$ konstant ist (er hängt nicht von $\v x$ ab), ist die Hessematrix die Nullmatrix. Wir schreiben auch $\nabla^2f(\v x)=\v H(\v x)=0$.

Hessematrix einer quadratischen Funktion 
: Jede quadratische Funktion lässt sich schreiben als 
$f(x_1,\dots,x_n)=\frac{1}{2}\v x^T\v A\v x+\v b^T\v x+c$. Da der Gradient $\nabla f(x_1,\dots,x_n)^T=\v A\v x+\v b$ eine lineare Funktion ist, ist die Hessematrix gleich der Matrix $\v A$, also $\nabla^2f(\v x)=\v H(\v x)=\v A$.

````


````{prf:theorem} Satz von Schwarz
Sei $f:\R^2\rightarrow\R$, $(x,y)\mapsto f(x,y)$ eine zwei mal stetig differenzierbare Funktion. Dann gilt:
\begin{align}
    \derv{}{y}\left(\derv{f}{x}\right) = \derv{}{x}\left(\derv{f}{y}\right).
\end{align}
````
Der Satz besagt, dass es egal ist in welcher Reihenfolge die partiellen Ableitungen gebildet werden. Das bedeutet, dass die Hessematrix symmetrisch ist.

````{prf:example}
Gegeben sei die Funktion $f:\R^2\rightarrow\R$, $f(x,y)=x^2\sin y$. Die ersten partiellen Ableitungen nach $x$ bzw. $y$ lauten:
\begin{align*}
    \derv{f}{x}=2x\sin y & &\derv{f}{y}=x^2\cos y
\end{align*}
Leiten wir nun beide Terme erneut ab, erhalten wir:
\begin{align*}
    \dervquad{f}{x}=2\sin y & & \dervquad{f}{y} = -x^2\sin y\\
    \dervzwei{f}{y}{x} = 2x\cos y & & \dervzwei{f}{x}{y} = 2x \cos y.
\end{align*}
````

Analog zur zweiten Ableitung im eindimensionalen beschreibt die Hessematrix die *Krümmung* einer Funktion.
````{prf:example}
Wir betrachten wieder das Paraboloid gegeben durch $f:\R^2\rightarrow\R$, $f(x,y)=x_1^2+x_2^2$. Als Hessematrix erhalten wir
\begin{align*}
    \nabla_{xy}^2 f= \bmat 
        2 & 0\\
        0 & 2
    \emat  
\end{align*}
Diese Matrix hat den (doppelten) *Eigenwert* $2$. Wenn eine quadratische Matrix nur positive Eigenwerte hat, nennt man sie *positiv definit*. Falls eine Hessematrix für alle Punkte $(x,y)$ aus dem Definitionsbereich positiv definit ist, nennt man die zugehörige Funktion *konvex*. In diesem Fall stimmt das mit der Anschauung überein, dass der Graph eines Paraboloids konvex ist. 
````
```{note}
Bildet man die zweite Ableitung einer *vektorwertigen* Funktion $\v f:\R^n\rightarrow\R^m$, erhält man "eine Hessematrix für jede Komponente $f_i$", also $m$ $n\times n$-Matrizen hintereinander---einen sog. $(m\times n\times n)$ *Tensor*.
```

(sec:richtung)=
## Richtungsableitungen
Anschaulich beschreiben die partiellen Ableitungen die Änderungsraten einer Funktion $f$ entlang der Koordinatenachsen (d.h. bei kleinen Variationen *einer* der Koordinaten). Wenn man sich nun nicht für die Änderung von $f$ in Richtung der Koordinatenachsen interessiert (das sagen einem die partiellen Ableitungen), sondern dafür, wie sich $f$ (ausgewertet an einem Punkt $\v x_0$) bei kleinen Änderungen in einer beliebigen *Richtung* ändert, so kann man mit Hilfe des Gradienten die *Richtungsableitung* von $f$ in einer beliebigen Richtung bestimmen.

Wir präzisieren zunächst, was die Richtungsableitung eigentlich genau aussagen soll. Dazu definieren wir einen *Richtungsvektor* $\v a\in \R^n$ als einen beliebigen Vektor der Länge $1$, also $\norm{\v a}_2=1$. Ähnlich wie bei der Definition der partiellen Ableitungen führen wir die Richtungsableitung wieder auf den eindimensionalen Fall zurück. Dazu konstruieren wir zunächst eine Gerade im $\R^n$, die durch $\v x_0$ geht und in die Richtung $a$ "zeigt": 
\begin{align*}
\v z(t)=\v x_0+t\v a, \quad t\in\R.
\end{align*}

Wir können nun die Funktion $f$ auf diese Gerade anwenden (sie ist ja eine Gerade im Definitionsbereich von $f$). Damit wird aus der Gerade im $\R^n$ eine Kurve im $\R^{n+1}$ die auf dem Funktionsgraphen von $f$ liegt:
\begin{align*}
f(\v z(t))=f(\v x_0+t\v a)
\end{align*}
Hier ein Beispiel für den Fall $f:\R^2\rightarrow \R, f(x,y)=(1-x)^2 + (y-x^2)^2$, $x_0(0,0)$ und $a=\frac{1}{\sqrt{5}}\bmat 1\\2\emat  $:
```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import numpy as np

# Die Funktion f als Surface Plot
x = np.linspace(-1.5,1.5,100)
y = np.linspace(-0.5,2.5,100)
X,Y = np.meshgrid(x,y)
z = (1-X)**2 + (Y-X**2)**2

fig = go.Figure(go.Surface(x=x,y=y,z=z, colorscale="Blues", showscale=False))

# Richtungsvektor a=(2,1)
a1 = 2
a2 = 1
# Argument t der Kurve läuft von -0.5 bis 0.75
t = np.linspace(-0.5,0.75,100)
x = t*a1
y = t*a2
z = (1-x)**2 + (y-x**2)**2

# x,y,z enthält jetzt die Koordinaten der Kurve f(0+t*a). 
fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode="lines", line=dict(width=5)))
fig.update_layout( autosize=True,
                  margin=go.layout.Margin(l=0, r=0, b=0, t=0))
```
Die Gerade $\v z(t)=t\bmat 1\\2\emat  $ in der Ebene wird mittels $f(\v z(t))$ eine Kurve im $\R^3$.

Die Richtungsableitung in Richtung $a$ ist nun die Ableitung der (univariaten) Funktion $g:\R\rightarrow\R$, $g(t):=f(\v z(t))=f(\v x_0+t\v a)$ an der Stelle $\v x_0$. Wie bekommen wir die? Natürlich mit der Kettenregel! Dies ist eine skalare Funktion, die einen "Umweg" über den $\R^n$ macht:
\begin{align*}
\R&\rightarrow \R^n&&\rightarrow \R\\
t&\mapsto \v x_0+t\v a=z &&\mapsto f(z)
\end{align*}
Anwendung der Kettenregel:
\begin{align*}
g'(t)&=\derv{g}{t}=\derv{f}{z}\derv{\v z}{t}\\
     &=\left(\derv{f}{z_1},\dots,\derv{f}{z_2})\right)\cdot \bmat a_1\\\dots\\a_n\emat  =\nabla f(\v z)\cdot \v a
\end{align*}
Ausgewertet an der Stelle $t=0$ ergibt sich:
\begin{align*}
g'(t)=\nabla f(\v x_0)\cdot \v a
\end{align*}
Das bedeutet: um die Änderungsrate der Funktion $f$ in einer bestimmten Richtung $\v a$ zu berechnen, bildet man einfach das *Skalarprodukt* zwischen dem Gradientenvektor und dem Richtungsvektor.


### Skalarprodukt und Richtungsableitung
Aus der linearen Algebra wissen wir, dass die Definition des Skalarproduktes im $\R^n$ lautet:
\begin{align*}
\nabla f\cdot \v a= \norm{\nabla f}\cdot \norm{\v a}\cdot \cos \varphi,
\end{align*}
wobei $\varphi$ der eingeschlossene Winkel zwischen dem Gradienten $\nabla f$ und der Richtung $\v a$ ist.

Daraus ergeben sich folgende beiden geometrischen Zusammenhänge:
1. Das Skalarprodukt, also die Änderungsrate der Funktion wird (betragsmäßig) *maximal*, wenn $\cos \varphi=1$, d.h. $\nabla f$ und $\v a$ sind parallel. Daraus folgt, dass der Gradientenvektor in Richtung des steilsten Anstiegs zeigt.
2. Das Skalarprodukt, also die Änderungsrate der Funktion wird (betragsmäßig) *minimal*, wenn $\cos \varphi=0$, d.h. $\nabla f$ und $\v a$ sind orthogonal. Der Gradient ist dann insgesamt $0$. Die zum Gradienten orthogonalen Richtungen sind diejenigen Richtungen, die entlange der Höhenlinien zeigen. Der Gradientenvektor in einem Punkt $\v x_0$ steht also senkrecht auf den Höhenlinien.

### Jacobimatrix und Richtungsableitung
Die Richtungsableitung kann man natürlich auch für Funktionen $f:\R^n\rightarrow \R^m$ berechnen. Hier erhält man dann einen $m$-dimensionalen Vektor als Richtungsableitung, wobei jeder Eintrag das Skalarprodukt einer Zeile der Jacobimatrix mit dem Richtungsvektor ist (das ist gerade das gewöhnliche Matrix-Vektor-Produkt):
\begin{align*}
\nabla f\cdot \v a&=\bmat \derv{f_1}{x_1} & \cdots & \derv{f_1}{x_n}\\
                                \vdots & &\vdots\\
                                \derv{f_m}{x_1}& \cdots & \derv{f_m}{x_n}\emat  
                                \bmat a_1 \\ \vdots \\a_n\emat  \\
                &=\bmat a_1\derv{f_1}{x_1}+ \cdots + a_n \derv{f_1}{x_n}\\
                                \vdots\\
                                a_1n\derv{f_m}{x_1}+ \cdots + a_n \derv{f_m}{x_n}\emat  \in\R^m
\end{align*} -->

(sec:interpretation)=
## Zusammenfassung: Interpretation des Gradienten als Änderungsrate einer Funktion
Wir haben in diesem Kapitel den Begriff der multivariaten Funktion und der partiellen Ableitung eingeführt. Als Anschauung diente uns dabei stets der Begriff der *Änderung* (genauer: der Änderungsrate) einer Funktion, die durch die partiellen Ableitung ausgedrückt werden. 
Wenn man sich Gradienten $\nabla f$ einer Funktion $f:\R^n\rightarrow \R$ in einem beliebigen Punkt $\v x_0$ vorstellen möchte, so haben wir folgende hilfreiche Anschauungen diskutiert:

- Die Einträge des Gradienten beschreiben die *Änderungsrate* des Funktionswerts $f$ bei kleinen[^fn:infinitesimal] Schritten in Richtung der Koordinatenachsen.
<!-- - Das Skalarprodukt eines Richtungsvektors $\v a$ der Länge 1 (also $||\v a||_2=1$) mit dem Gradienten, $\nabla f \cdot v$ beschreibt die Änderungsrate der Funktion in einer Richtung (wie ändert sich Funktion, wenn man das Argument ein kleines[^fn:infinitesimal] Stück in Richtung des Richtungsvektors $\v a$ verschiebt). Das funktioniert auch mit der Jacobimatrix. -->
- Geometrisch: Der Gradient ist ein Vektor, der in Richtung des steilsten Anstiegs zeigt.
- Geometrisch: Der Gradient ist ein Vektor, der senkrecht auf den Höhenlinien steht. 

Eine weitere Interpretation ist die folgende: 

- Der Gradient bzw. die Jacobi Matrix ausgewertet an einem Punkt können als lineare Abbildung aufgefasst werden, die die Funktion in einer Umgebung dieses Punktes approximieren. Ableitungen können also als *Approximation* einer differenzierbaren Funktion betrachtet werden. 

Wichtig: es handelt sich dabei nur um unterschiedliche Sichtweisen auf ein und dasselbe Konzept ("Ableitung"). Keine der verschiedenen Interpretationen von Ableitungen ist per se besser oder schlechter. Alle sind richtig. Warum gebe ich dem Begriff der Ableitung überhaupt so viel Raum in dieser Vorlesung? Funktionen, insbesondere multivariate, sind komplizierte Objekte, für die man nur sehr schwer eine Anschauung entwickeln kann. Das geht manchmal unter, wenn man sich nur mit einfachen Beispielen in einer oder zwei Dimensionen beschäftigt.
Im maschinellen Lernen versuchen wir beispielsweise extrem komplizierte mathematische Funktionen zu lernen, die sehr komplizierte Sachverhalte nachbilden ("Email-Text $\rightarrow$ {Spam, kein Spam}", "historische Daten$\rightarrow$ zukünftiger Umsatz", "Bild$\rightarrow$ Objekte im Bild"). Es ist hoffnungslos, sich eine solche Funktion vorzustellen. Wenn wir aber die Ableitung einer Funktion kennen, so können wir zumindest approximieren, wie sich die Funktion in der Umgebung eines Punktes verhält -- nämlich wie eine lineare Funktion, was wiederum die einfachste Klasse von Funktionen ist (lineare Algebra!). Algorithmen im maschinellen Lernen bedienen sich dieser zusätzlichen Information. 

%Nächstes Kapitel: über Approximation, Taylorreihen (als alternative Interpretation, statt über Änderungen), totale Differenzierbarkeit.

[^fn:infinitesimal]: Eigentlich: *infinitesimal*, also "unendlich" kleine Änderungen - was natürlich schon wieder weniger anschaulich ist. 

