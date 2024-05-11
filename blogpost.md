# Title

**Authors:** *S.A. Abbring, H.C. van den Bos, R. den Braber, A.J. van Breda, D. Zegveld*

---

TODO: Introduction text

---

## Color for Classification

<!---
TODO: Text about the influence of color on classification (related work: color invariance)
--->
Color is an important feature for recognition/classification by humans. One example is investiged by \[1\], who found that color facilitates expert bird-watchers in faster and more accurate recognition at both high (family) and low (specimen) levels of bird recognition. 

## Recap on Group Equivariant Convolutions

<!---
TODO: Explain Group Equivariant Convolutions (technical)
--->
Deep Convolutional Neural Networks have been around since the late 1980s and over the years proven to be highly effective for image classification \[1\]. Empirical evidence shows the importance of depth for good performance and convolutional weight-sharing for parameter reduction. The latter is effective due to the translation symmetry inherent in most image data, whereby the data is roughly invariant to shifts, such that the same weights can be utilised to convolve different parts of the image \[2\]. Convolution layers are translation equivariant in a deep network: the output shifts relative to shifts in the input. This notion of symmetry can be extended to larger groups, including rotation.

The generalization of translation equivariance is achieved through Group Convolutional Neural Networks (G-CNN). A CNN layer is equivariant to a group if for all transformations $g \in G$, doing the transformation $T_g$ on the input and then the feature mapping $\Phi (x)$ is similar to doing the feature mapping on the input and the transformation $T'_g$ thereafter: 

$$\begin{align} 
\Phi (T_g x) = T'_g \Phi (x) & \qquad \qquad \forall g \in G, & \qquad \qquad (\text{Equation 1})\\ 
\end{align}$$

where $T_g$ and $T'_g$ can be equivalent.
We utilise the equation from \[2\] to show that G-CNNs are equivariant. Instead of shifting a filter, correlation in the first layer can be described more generally by replacing it with some transformation from group $G$, whereby $f$ is the input image and $\psi$ is the filter:

$$ 
[f \star \psi](g) = \sum_{y \in \mathbb{Z}^2}\sum_k f_k(y)\psi_k(g^{-1}y) \qquad \qquad (\text{Equation 2})
$$

$$[f \star \psi](g) = \sum_{y \in \mathbb{Z}^2}\sum_k f_k(y)\psi_k(g^{-1}y) \qquad \qquad (\text{Equation 2})$$

$$\begin{align} 
[f \star \psi](g) = \sum_{y \in \mathbb{Z}^2}\sum_k f_k(y)\psi_k(g^{-1}y) & \qquad \qquad \text{(Equation 4)} \\ 
\end{align}$$

Since the feature map $f \star \psi$ is a function on G, the filters are functions on G for all layers after the first. The correlation then becomes:

$$\begin{align} 
[f \star \psi](g) = \sum_{h \in G}\sum_k f_k(h)\psi_k(g^{-1}h) & \qquad \qquad (\text{Equation 3})\\ 
\end{align}$$

Using the substition $h \rightarrow uh$, the equivariance of the correlation can be derived such that a translation followed by a correlation is equivalent to a correlation followed by a translation:

$$\begin{align} 
[[L_uf] \star \psi](g) &= \sum_{h \in G}\sum_k f_k(u^{-1}h)\psi(g^{-1}h)\\ 
&= \sum_{h \in G}\sum_kf(h)\psi(g^{-1}uh)\\
&= \sum_{h \in G}\sum_kf(h)\psi((u^{-1}g)^{-1}h)\\
&= [L_u[f \star \psi]](g) & \qquad \qquad (\text{Equation 4})\\
\end{align}$$

<!---
Mss zijn deze formules allemaal net iets teveel overgenomen van [2]
--->

## Color Equivariance

TODO: explain specifically for color equivariance (technical)

## (Maybe Architecture/Evaluation/Dataset(s) explanation or Something)

## Reproduction of Experiments

TODO: explain findings about the reproduction of figure 2, figure 2, figure 9 and figure 13 in the following narrative: 

### When is color equivariance useful? 

#### Color imbalance

TODO: (figure 2 about color imbalance) 

#### Color Selectivity

TODO: in which stages is color equivariance useful (figure 3 about color selective datasets)

### Color Equivariance in Image Classification and impact of hyperparameters

#### Image Classification

TODO: color equivariant cnns versus vanilla cnns (table 1 but then figure 9)

#### Number of Rotations

TODO: the impact of the number of hue rotations (figure 13)

## Further Research

### Rens

### Dante

### Silvia

TODO: create a nice narrative with these three

## Concluding Remarks

## Authors' Contributions

## References
<a id="1">[1]</a> 
W. Rawat and Z. Wang, "Deep Convolutional Neural Networks for Image Classification: A Comprehensive Review," in Neural Computation, vol. 29, no. 9, pp. 2352-2449, Sept. 2017, doi: 10.1162/neco_a_00990. 

<a id="1">[2]</a>
Cohen, T. &amp; Welling, M.. (2016). Group Equivariant Convolutional Networks. <i>Proceedings of The 33rd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 48:2990-2999 Available from https://proceedings.mlr.press/v48/cohenc16.html.


