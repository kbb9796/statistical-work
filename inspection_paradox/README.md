The inspection paradox is one of my favorite results in probability
theory. It says that for a renewal process with a finite average waiting
time, an observer will spend more time waiting for the next arrival than
the average waiting time itself. For example, if a bus arrives at a bus
stop every 5 minutes on average, someone waiting to get on the bus will
wait longer than 5 minutes on average. This is certainly paradoxical,
but nonetheless feels true from experience. I prove this fact below.

::: {.proof}
*Proof.* To start, we define some variables. Let $X_t$ be the value of
the renewal process at time $t$; let $J_{X_t}$ be the time of the
$X_t$'th arrival, and $S_{X_t}$ the holding time for the $X_t$'th
arrival. Now, consider time $t$ in the interval
$[J_{X_t}, J_{{X_t}+1}]$, i.e. we have observed $X_t$ but not $X_t + 1$.
Then, we have $$\begin{aligned}
    P(S_{{X_t}+1} > x) = \int_{0}^{\inf} P(S_{{X_t}+1} > x | J_{X_t} = s) f_{J_{X_t}} ds \\
    &= \int_{0}^{\inf} P(S_{{X_t}+1} > x | S_{{X_t}+1} > t - s) f_{J_{X_t}} ds \\
    &= \int_{0}^{\inf} \frac{P(S_{{X_t}+1} > x, S_{{X_t}+1} > t - s)}{P(S_{{X_t}+1} > t - s)} f_{J_{X_t}} ds \\
    &= \int_{0}^{\inf} \frac{1 - F(max{x, t -s})}{1 - F(t -s)} f_{J_{X_t}} ds \\
    &= \int_{0}^{\inf} min{\frac{1 - F(x)}{1 - F(t - s)}, \frac{1 - F(t -s )}{1 - F(t - s)}} f_{J_{X_t}} ds \\
    &= \int_{0}^{\inf} min{\frac{1 - F(x)}{1 - F(t - s)}, 1} f_{J_{X_t}} ds \\
    &\geq \int_{0}^{\inf} (1 - F(x)) f_{J_{X_t}} ds \\
    &= 1 - F(x)
    &= P(S_{1} > x)
  \end{aligned}$$ ◻
:::
