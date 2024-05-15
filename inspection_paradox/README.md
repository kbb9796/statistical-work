To start, we define some variables. Let $X_t$ be the value of the renewal process at time $t$; let $J_{X_t}$ be the time of the $X_t$'th arrival, 
and $S_{X_t}$ the holding time for the $X_t$'th arrival. Now, consider time $t$ in the interval $[J_{X_t}, J_{{X_t}+1}]$, i.e. we have observed 
$X_t$ but not $X_t + 1$. Then, we have

$$
P(S_{{X_t}+1} > x) = \int_{0}^{\inf} P(S_{{X_t}+1} > x | J_{X_t} = s) f_{J_{X_t}}(s) ds \\
&= \int_{0}^{\infty} P(S_{{X_t}+1} > x | S_{{X_t}+1} > t - s) f_{J_{X_t}}(s) ds \\
&= \int_{0}^{\infty} \frac{P(S_{{X_t}+1} > x, S_{{X_t}+1} > t - s)}{P(S_{{X_t}+1} > t - s)} f_{J_{X_t}}(s) ds \\
&= \int_{0}^{\infty} \frac{1 - F(max{x, t -s})}{1 - F(t -s)} f_{J_{X_t}}(s) ds \\
&= \int_{0}^{\infty} min{\frac{1 - F(x)}{1 - F(t - s)}, \frac{1 - F(t -s )}{1 - F(t - s)}} f_{J_{X_t}}(s) ds \\
&= \int_{0}^{\infty} min{\frac{1 - F(x)}{1 - F(t - s)}, 1} f_{J_{X_t}}(s) ds \\
&\geq \int_{0}^{\infty} (1 - F(x)) f_{J_{X_t}}(s) ds \\
&= 1 - F(x)
&= P(S_{1} > x)
$$
