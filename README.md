# tf-Gaussian-distribution-approximation-layer

value representation
-----------------------------------
V = (μ, s)

V with Average μ, Variance s

add 
-----------------------------------
![(\mu_{1}, s_{1}) + (\mu_{2}, s_{2}) = (\mu_{1} + \mu_{2}, s_{1} + s_{2})](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%28%5Cmu_%7B1%7D%2C+s_%7B1%7D%29+%2B+%28%5Cmu_%7B2%7D%2C+s_%7B2%7D%29+%3D+%28%5Cmu_%7B1%7D+%2B+%5Cmu_%7B2%7D%2C+s_%7B1%7D+%2B+s_%7B2%7D%29)

multiple 
-----------------------------------
![(\mu_{1}, s_{1}) \times (\mu_{2}, s_{2}) = (\mu_{1} \times \mu_{2}, (\mu_{1}^2 + s_{1}) (\mu_{2}^2 + s_{2}) - \mu_{1}^2 \mu_{2}^2](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%28%5Cmu_%7B1%7D%2C+s_%7B1%7D%29+%5Ctimes+%28%5Cmu_%7B2%7D%2C+s_%7B2%7D%29+%3D+%28%5Cmu_%7B1%7D+%5Ctimes+%5Cmu_%7B2%7D%2C+%28%5Cmu_%7B1%7D%5E2+%2B+s_%7B1%7D%29+%28%5Cmu_%7B2%7D%5E2+%2B+s_%7B2%7D%29+-+%5Cmu_%7B1%7D%5E2+%5Cmu_%7B2%7D%5E2)


activation-ReLU
---------------------------------
> calculate based on {X∈(μ, s)|X≥0}

p = P( {X|X∈(μ, s)} ≥ 0)

![EC1 = \int_{-\frac{\mu}{\sqrt{s}}}^{\infty } \frac{x}{\exp(\frac{x^2}{2}) \sqrt{2 \pi}} dx](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+EC1+%3D+%5Cint_%7B-%5Cfrac%7B%5Cmu%7D%7B%5Csqrt%7Bs%7D%7D%7D%5E%7B%5Cinfty+%7D+%5Cfrac%7Bx%7D%7B%5Cexp%28%5Cfrac%7Bx%5E2%7D%7B2%7D%29+%5Csqrt%7B2+%5Cpi%7D%7D+dx)

![EC2 = \int_{-\frac{\mu}{\sqrt{s}}}^{\infty } \frac{x^2}{\exp(\frac{x^2}{2}) \sqrt{2 \pi}} dx](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+EC2+%3D+%5Cint_%7B-%5Cfrac%7B%5Cmu%7D%7B%5Csqrt%7Bs%7D%7D%7D%5E%7B%5Cinfty+%7D+%5Cfrac%7Bx%5E2%7D%7B%5Cexp%28%5Cfrac%7Bx%5E2%7D%7B2%7D%29+%5Csqrt%7B2+%5Cpi%7D%7D+dx)

![(\mu_{1}, s_{1})\xrightarrow{relu}{(\mu p + EC1, EC2 - EC1^2)}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%28%5Cmu_%7B1%7D%2C+s_%7B1%7D%29%5Cxrightarrow%7Brelu%7D%7B%28%5Cmu+p+%2B+EC1%2C+EC2+-+EC1%5E2%29%7D)

activation-Softmax
---------------------------------

![E(\exp(N(\mu, s))) = \exp(\mu + s/2)](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+E%28%5Cexp%28N%28%5Cmu%2C+s%29%29%29+%3D+%5Cexp%28%5Cmu+%2B+s%2F2%29)

![V(\exp(N(\mu, s))) = E(\exp(N(\mu, s))^2) - E(\exp(N(\mu, s)))^2](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V%28%5Cexp%28N%28%5Cmu%2C+s%29%29%29+%3D+E%28%5Cexp%28N%28%5Cmu%2C+s%29%29%5E2%29+-+E%28%5Cexp%28N%28%5Cmu%2C+s%29%29%29%5E2)

![G_\mu = \sum E(\exp(N(\mu_i, s_i)))](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+G_%5Cmu+%3D+%5Csum+E%28%5Cexp%28N%28%5Cmu_i%2C+s_i%29%29%29)

![(\mu_i, s_i) \xrightarrow{softmax} {(\frac{E(\exp(N(\mu_i, s_i)))}{G_\mu}, \frac{V(\exp(N(\mu_i, s_i)))}{G_\mu})}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%28%5Cmu_i%2C+s_i%29+%5Cxrightarrow%7Bsoftmax%7D+%7B%28%5Cfrac%7BE%28%5Cexp%28N%28%5Cmu_i%2C+s_i%29%29%29%7D%7BG_%5Cmu%7D%2C+%5Cfrac%7BV%28%5Cexp%28N%28%5Cmu_i%2C+s_i%29%29%29%7D%7BG_%5Cmu%7D%29%7D)

activation-Sigmoid
----------------------------------
![\begin{equation}
\int_x sigmoid(x) \mathcal{N}(x; \mu, \sigma^2) \approx \int_x \Phi(\lambda x) \mathcal{N}(x; \mu, \sigma^2) = \Phi\left(\frac{\lambda \mu}{\sqrt{1 + \lambda^2 \sigma^2}}\right).
\end{equation}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%7D%0A%5Cint_x+sigmoid%28x%29+%5Cmathcal%7BN%7D%28x%3B+%5Cmu%2C+%5Csigma%5E2%29+%5Capprox+%5Cint_x+%5CPhi%28%5Clambda+x%29+%5Cmathcal%7BN%7D%28x%3B+%5Cmu%2C+%5Csigma%5E2%29+%3D+%5CPhi%5Cleft%28%5Cfrac%7B%5Clambda+%5Cmu%7D%7B%5Csqrt%7B1+%2B+%5Clambda%5E2+%5Csigma%5E2%7D%7D%5Cright%29.%0A%5Cend%7Bequation%7D)

loss 
-------------------------------------

traditional basic loss = mse

Gaussian approximation basic loss = Negative log likelihood of gaussian distribution probability density function



initialization
--------------------------------------
weight  = (0, s)

bias    = (0, 0)




<!--make LaTeX code to URL from https://tex-image-link-generator.herokuapp.com/-->
## References
<a id="1">[1]</a> 
@MISC {1994805,
    TITLE = {Expected value of applying the sigmoid function to a normal distribution},
    AUTHOR = {Smoke (https://math.stackexchange.com/users/377538/smoke)},
    HOWPUBLISHED = {Mathematics Stack Exchange},
    NOTE = {URL:https://math.stackexchange.com/q/1994805 (version: 2016-11-01)},
    EPRINT = {https://math.stackexchange.com/q/1994805},
    URL = {https://math.stackexchange.com/q/1994805}
}
