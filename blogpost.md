# Title

**Authors:** *S.R. Abbring, H.C. van den Bos, R. den Braber, A.J. van Breda, D. Zegveld*

---

<!--
TODO: Introduction text
--->
In this blogpost, we discuss, analyze and extend upon the findings of the paper titled *Color Equivariant Convolutional Networks*\[main\]. The paper introduces Color Equivariant Convolutions (CEConvs) by leveraging parameter sharing over hue-shifts. The authors demonstrate the benefits of the novel model in terms of robustness to color alterations and accuracy performance.
The objectives of this blogpost are to:

1. Discuss the methods introduced in the paper
1. Verify the authors' claims
1. Extend the notion of color equivariance to other dimensions beyond hue

---

## Introduction

<!---
TODO: Text about the influence of color on classification (related work: color invariance)
--->
Color is a crucial feature for recognition and classification by humans. For example, a study by \[bird\] found that color facilitates expert bird-watchers in faster and more accurate recognition at both high (family) and low (specimen) levels of bird identification. Similarly, the convolutional layers in a Convolutional Neural Network (CNN) exhibit color representation akin to the human vision system \[human_vision\] with all layers containing color selective neurons. These color representations are present at three different levels: in single neurons, in double neurons for edge detection and in combination with shape at all levels in the network.

Although color invariance has been achieved in various research areas, such as in facial recognition to mitigate the influence of lightning conditions \[color_invariance\], some classification problems are color dependent. Therefore, instead of training CNNs to classify images regardless of their color (invariance), it might be more beneficial to classify images based on their color (equivariance). 

The Color Equivariant Convolutions (CEConvs) introduced in \[main\] achieve this through equivariance to discrete hue shifts. The hue is represented in RGB space such that it can be expressed as a 3D rotation around the [1, 1, 1] axis. This approach utilizes group convolutions as introduced by \[group_convs\] which can be equivariant to 3D rotations. We reproduce the results showing the effects of the CEConvs on color-imbalanced and color-selective datasets, as well as their impact on image classification. We examine the ablation studies to understand the impact of data-augmentation and rotation on the CEConvs, providing additional insights into computational requirements as well. Finally, we extend the notion of color equivariance beyond hue shifts.

The most significant limitation of the CEConvs is that color equivariance is modeled as solely equivariance to hue shifts. By extending the notion to other dimensions, such as saturation equivariance, the CNN could achieve a higher level of equivariance, as saturation can handle a greater variety of changes in illumination. Additionally, by modeling hue shifts as 2D rotations compared to 3D rotations for the RGB space, we circumvent the limitation described in \[main\]. This limitation involves pixel values falling outside the RGB cube, requiring a reprojection operation and consequently only allowing for an approximation of hue equivariance for pixels near the border of the RGB cube.

## Recap on Group Equivariant Convolutions

<!---
TODO: Explain Group Equivariant Convolutions (technical)
--->
Deep Convolutional Neural Networks have been around since the late 1980s and over the years proven to be highly effective for image classification \[DCNN\]. Empirical evidence shows the importance of depth for good performance and convolutional weight-sharing for parameter reduction. The latter is effective due to the translation symmetry inherent in most image data, whereby the data is roughly invariant to shifts, such that the same weights can be utilised to convolve different parts of the image \[group_convs\]. Convolution layers are translation equivariant in a deep network: the output shifts relative to shifts in the input. This notion of symmetry can be extended to larger groups, including rotation.

This generalization of translation equivariance is achieved through Group Convolutional Neural Networks (G-CNN). A CNN layer is equivariant to a group if for all transformations $g \in G$, doing the transformation $T_g$ on the input and then the feature mapping $\Phi (x)$ is similar to doing the feature mapping on the input and the transformation $T'_g$ thereafter: 

$$\begin{align*} 
\Phi (T_g x) = T'_g \Phi (x) & \qquad \qquad \forall g \in G, & \qquad \qquad (\text{Equation 1})
\end{align*}$$

where $T_g$ and $T'_g$ can be equivalent.
We utilise the equations from \[group_convs\] to show that G-CNNs are equivariant. Instead of shifting a filter, correlation in the first layer can be described more generally by replacing it with some transformation from group $G$, whereby $f$ is the input image and $\psi$ is the filter:

$$\begin{align*} 
\[ f \star \psi \](g) = \sum_{y \in \mathbb{Z}^2}\sum_{k} f_k(y) \psi_{k}(g^{-1}y) & \qquad \qquad (\text{Equation 2})
\end{align*}$$

Since the feature map $f \star \psi$ is a function on G, the filters are functions on G for all layers after the first. The correlation then becomes:

$$\begin{align} 
\[ f \star \psi \](g) = \sum_{h \in G}\sum_{k}f_k(h)\psi_{k}(g^{-1}h) & \qquad \qquad (\text{Equation 3})\\ 
\end{align}$$

Using the substition $h \rightarrow uh$ and the notation:
$$\begin{align} \[ L_gf \](x) = [f \circ g^{-1}](x) = f(g^{-1}x) & \qquad \qquad (\text{Equation 4})\end{align} $$

, the equivariance of the correlation can be derived such that a translation followed by a correlation is equivalent to a correlation followed by a translation:

$$\begin{align} 
\[\[L_uf\] \star \psi\](g) &= \sum_{h \in G}\sum_k f_k(u^{-1}h)\psi(g^{-1}h)\\ 
&= \sum_{h \in G}\sum_kf(h)\psi(g^{-1}uh)\\
&= \sum_{h \in G}\sum_kf(h)\psi((u^{-1}g)^{-1}h)\\
&= \[L_u\[f \star \psi\]\](g) & \qquad \qquad (\text{Equation 5})\\
\end{align}$$

<!---
Mss zijn deze formules allemaal net iets teveel overgenomen van [2]
--->

## Color Equivariance
<!---
TODO: explain specifically for color equivariance (technical)
--->
The original paper exploits the concept of group equivariant convolutions to achieve color equivariance, defined as equivariance to hue shifts. In the HSV (Hue-Saturation-Value) color space, hue is represented as an angular scalar value. The hue value is shifted by adding an offset after which the modulo is taken to ensure a valid range. The HSV space is reprojected to the RGB (Red-Green-Blue) color space such that the hue shifts correspond to a rotation along the diagonal vector [1, 1, 1]. 

This definition is extended to group theory, by defining the group $H_n$ as a subgroup of the $SO(3)$ group. Specifically, $H_n$ consists of multiples of 360/n-degree rotations about the [1, 1, 1] diagonal vector in $\mathbb{R}^3$ space. The rotation around a unit vector $\mathbf{u}$ by angle $\theta$ is defined in 5 steps: 

1. Rotate the vector such that it lies in one of the coordinate planes (e.g. $xz$)
1. Rotate the vector such that it lies on one of the coordinate axes (e.g. $x$)
1. Rotate the point around vector $\mathbf{u}$ on the x-axis
1. Reverse the rotation in step 2
1. Reverse the rotation in step 1

This leads to the following parameterization of $H_n$, with $n$ the number of rotations (discrete) and $k$ the rotation:

$$ 
H_n = 
\begin{bmatrix}
\cos (\frac{2k\pi}{n}) + a & a - b & a + b \\
a + b & \cos (\frac{2k\pi}{n}) + a & a - b \\
a - b & a + b & \cos (\frac{2k\pi}{n}) + a \\
\end{bmatrix}
$$

The group of discrete hue shifts is combined with the group of discrete 2D translations into the group $G = \mathbb{Z}^2 \times H_n$. Now Color Equivariant Convolution (CEConv) in the first layer is defined:

$$\begin{align} 
\[f \star \psi^i\](x, k) = \sum_{y \in \mathbb{Z}^2}\sum_{c=1}^{C^l}f_c(y) \cdot H_n(k)\psi_c^i(y - x) & \qquad \qquad (\text{Equation 6})\\ 
\end{align}$$

For the derivation of the equivariance of the CEConv layer, we refer to the original paper \[main\].

For the hidden layers, the feature map $[f \star \psi]$ is a function on $G$ parameterized by x,k. The CEConv hidden layers are defined as:

$$\begin{align} 
\[f \star \psi^i\](x, k) = \sum_{y \in \mathbb{Z}^2}\sum_{r=1}^n\sum_{c=1}^{C^l}f_c(y,r) \cdot \psi_c^i(y - x, (r-k)\%n) & \qquad \qquad (\text{Equation 7})\\ 
\end{align}$$

<!---
The operator $\mathcal{L}_g = \mathcal{L}_{(t, m)}$ expresses the translation $t$ and hue shift $m$ acting on input $f$:

$$\begin{align} 
[\mathcal{L}_gf](x) = [\mathcal{L}_{(t,m)}f](x) = H_n(m)f(x-t)& \qquad \qquad (\text{Equation 7})\\ 
\end{align}$$

The derivation equivariance of the CEConv layer can be derived (for $C^l = 1$) as:
--->


## (Maybe Architecture/Evaluation/Dataset(s) explanation or Something)

## Reproduction of Experiments

<!---
TODO: explain findings about the reproduction of figure 2, figure 2, figure 9 and figure 13 in the following narrative: 
--->
The reproduction of (a selection of) the experiments is primarily achieved through the code provided along with the original paper. However, it does not include the code to reproduce the plots which had to be written manually. Moreover, supplementing functionalities such as saving, loading and evaluation of the results needed to be integraded for sufficient reproduction. Lastly, not all commands are provided/explained in the READme file to easily run all experiments. Therefore, some investigation of the code was needed to run the exact experiments.

### When is color equivariance useful? 

Firstly, the experiments that show the importance of color equivariance are reproduced. This mainly includes exploring various dataset, starting with a color imbalanced dataset and followed by the investigation of datasets with a high/low color selectivity.

#### Color imbalance

To verify that color equivariance can share shape information across classes, we reproduced the long-tailed ColorMNIST experiment.  In this experiment, a 30-way classification is performed on a power law distributed dataset where 10 shapes (digits 0-9) and 3 colors (Red, Green, Blue) need to be distinguished. During training, classes are not equally distributed. During testing, all classes are evaluated on 250 examples. Sharing shape information across colors is beneficial during this experiment as a certain digit may occur more frequently in one color than in another. 

Two models were tested. The Z2CNN, a vanilla CNN model, consists of 25,990 trainable parameters whereas the CECNN model consists of 25,207 trainable parameters, This is because the width of the CECNN is smaller. This is to ensure that the same amount of GPU memory is required to train the models, which was a priority of the original authors to have a level comparison. However, the training time of the two models differed significantly with the Z2CNN model training 59% $\pm$ 4 faster than the CECNN network. The exact training method and performance can be seen in the provided notebook. 

<!-- ![Longtailed dataset results](blogpost_imgs/Longtailed.png) -->
<div align="center">
  <img src="blogpost_imgs/Longtailed.png" alt="Longtailed dataset results" width="600">

  *Figure 1: ...*
</div>

The figure is ordered in the availability of training samples for every class. Performance of the shape-sharing CECNN consistently outperforms the baseline Z2CNN, where the average performance of the Z2CNN is 68.8% $\pm$ 0.6% and for the CECNN is 85.2 $\pm$ 1.2%. Most performance increase is seen in classes where little training data is provided thus confirming the hypothesis that the CECNN is able to share shape weight information effectively. These results are in line with the findings of the original authors which also describe a large performance increase. A difference in findings is the std of the CECNN which is larger than that of the Z2CNN however, this could be due to the randomness in data generation* which resulted in a different data distribution for our experiment.

<span style="color:grey">* We made the data generation deterministic by setting a seed, recreating our experiment would return the same data distribution.</span>


#### Color Selectivity
<!---
TODO: in which stages is color equivariance useful (figure 3 about color selective datasets)
--->
Color selectivity is defined as: “Color selectivity is the property of a neuron that activates highly when a particular color appears in the input image and, in contrast, shows low activation when this color is not present.” \[color_selectivity\]. The authors of the original paper utilize this notion to define color selectivity of a dataset. Namely, they computed the color selectivity as an average of all neurons in the baseline CNN trained on the respective dataset. We reproduced the experiment to investigate the influence of using color equivariance up to late stages. Due to computational constraints, only two of the four datasets were explored; flowers102 with the highest color selectivity (0.70) and STL10 with the lowest color selectivity (0.38). While we did not explore the remaining datasets extensively, their color selectivity was comparable to STL10, suggesting that our findings are inclusive for the additional datasets.

In Figure 2, the accuracy improvement of color equivariance up to later stages in the network are displayed for both mentioned datasets. The baseline is the ResNet18 model with one rotation (equivariance up to 0 stages). For the other values, HybridResNet18 models are trained with 3 rotations, max pooling, separable kernels and the number of color equivariant stages as shown in the figure. Additionally, the graph on the right shows the result with color-jitter augmentation.

<!-- ![Color selectivity results](blogpost_imgs/color_selectivity.png) -->
<div align="center">
  <img src="blogpost_imgs/color_selectivity.png" alt="Color selectivity results" width="600">

  *Figure 2: Influence of color equivariance embedded up to late stages in the network on datasets with high and low color selectivity.*
</div>

Similar to the original paper’s results, the color selective dataset seems to benefit from color equivariance up to later stages in the network, in contrast to the less color selective dataset. This is especially clear for the graph with color-jitter augmentation. However, the color selectivity seems detrimental at the earlier stages without color-jitter augmentation for the color selective dataset. In general, the accuracy improvements/deteriorations are less extreme compared to the original results. The differences might be explained by the fact that we trained the model on the full datasets instead of on a subset. By our results, we suspect that color equivariance is solely significantly beneficial for color selective datasets in combination with color-jitter augmentation. Otherwise, the differences are negligible. 

### Color Equivariance in Image Classification and impact of hyperparameters

We will now explore the reproduction of a variation on the main results along with a small insight into the hyperparameters. These results are all limited to the Flowers102 dataset since it has the largest color discreptency and the ResNet18 model, aligning with the original paper. The results were placed in the appendix of the original paper. However, we decided that the reproduction of the figure on one dataset is more insightful than an enormous table. The final experiment is an ablation study investigating the impact of varying the number of rotations. This aspect is altered across different experiments, highlighting its importance and deserving notice.

#### Image Classification

In our evaluation of image classification performance, we utilized the flowers-102 dataset due to its most prominent color dependency across the datasets evaluated by the original authors. Our study involved training a baseline ResNet-18 model comprising approximately 11,390,000 parameters, alongside the novel color equivariant CE-ResNet trained with three rotations. Both models underwent training with and without jitter, augmenting the training data with varying hue-intensity images. Subsequently, we assessed their performance on test sets subjected to gradual hue shifts ranging from -180° to 180°.

<!-- ![Classification Test-time Hue Shifts](blogpost_imgs/Test-time_Hue_Shifts.png) -->

<div align="center">
  <img src="blogpost_imgs/Test-time_Hue_Shifts.png" alt="Classification Test-time Hue Shifts" width="600">

  *Figure 3: ...*
</div>

In the figure presented, both the baseline ResNet and the CE-ResNet demonstrate good performance when no hue shift is applied (Test-time hue shift = 0). The CE-ResNet displays optimal performance in three specific hue areas, which reflects the orientations it is trained on. Moreover, the CE-ResNet consistently maintains performance levels above or equal to the original ResNet across almost all hue shifts, indicating its dominance across distributional changes.

When trained with jitter, both models exhibit robustness against distributional shifts, in line with the original authors findings, with the CE-ResNet-18 showing slightly better performance. This advantage is attributed to more efficiency in weight sharing, entailing more information can possibly be stored about other features. These models did take around 6 times as long to train than the non-jittered models. The extended training duration of these models can be attributed to the convoluted sampling process involved in generating jittered images.

Comparing training and testing times, the baseline model completes its training approximately 50% faster than the CEConv model. Testing time took around 2.3 times as long for the novel CEConv model. This indicates a significant speed advantage for production with the baseline model, albeit with a slight sacrifice in performance due to the non-utilization of CEConv.

#### Number of Rotations

<!---
TODO: the impact of the number of hue rotations (figure 13)
--->
The main implementation of the color-equivariance consists of adding three rotations of 120 degrees and the baseline model (not-equivariant) can be expressed as having 1 rotation. In Figure 4, we reproduced the experiments examining what happens with additional rotations. In order to save computational power, we limited the experiments to 1, 5 and 10 rotations (instead of 1-10 in the original paper). Nonetheless, the trends are the same.

<div align="center">
  <img src="blogpost_imgs/rotations.png" alt="Hue rotations" width="600">

  *Figure 4: Accuracy with varying rotations.*
</div>

The lines in the plot are not smooth because it has only been evaluated on 37 points. Nonetheless, the trends are similar to the original paper’s findings. The number of peaks aligns with the number rotations, additionally the height of the peaks decreases as the number of rotations increases. However, the peaks have different heights which might be attributed to the reprojection into the RGB cube range. Based on these results it seems that more rotations lead to higher equivariance. These results lead to a trade-off between the amount of equivariance, the maximum accuracy and the number of parameters as displayed in Table 1.

<center>

| Number of Rotations        | Number of Parameters     | Max. Accuracy    |
|--------------|-----------|-----------|
| 1 | 11.2 M  | 70.3%  |
| 5 | 11.6 M  | 72.4%  |
| 10 | 11.8 M  | 74.6%  |

*Table 1: Parameter and maximum accuracy increase based on number of rotations.*
</center>

#### Jitter


## Further Research

### Rens

### Dante

### Silvia

TODO: create a nice narrative with these three

## Concluding Remarks

## Authors' Contributions

## References
<a id="1">[bird]</a> 
Simen Hagen, Quoc C. Vuong, Lisa S. Scott, Tim Curran, James W. Tanaka; The role of color in expert object recognition. Journal of Vision 2014;14(9):9. https://doi.org/10.1167/14.9.9.

<a id="1">[human_vision]</a> 
Ivet Rafegas, Maria Vanrell; Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2697-2705 

<a id="1">[color_invariance]</a> 
R. Rama Varior, G. Wang, J. Lu and T. Liu, "Learning Invariant Color Features for Person Reidentification," in IEEE Transactions on Image Processing, vol. 25, no. 7, pp. 3395-3410, July 2016, doi: 10.1109/TIP.2016.2531280.
keywords: {Image color analysis;Lighting;Cameras;Histograms;Shape;Dictionaries;Robustness;Person re-identification;Illumination invariance;Photometric invariance;Color features;Joint learning;Person re-identification;illumination invariance;photometric invariance;color features;joint learning},

<a id="1">[DCNN]</a>
W. Rawat and Z. Wang, "Deep Convolutional Neural Networks for Image Classification: A Comprehensive Review," in Neural Computation, vol. 29, no. 9, pp. 2352-2449, Sept. 2017, doi: 10.1162/neco_a_00990. 

<a id="1">[group_convs]</a>
Cohen, T. &amp; Welling, M.. (2016). Group Equivariant Convolutional Networks. <i>Proceedings of The 33rd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 48:2990-2999 Available from https://proceedings.mlr.press/v48/cohenc16.html.

<a id="1">[main]</a>
Lengyel, A., Strafforello, O., Bruintjes, R. J., Gielisse, A., & van Gemert, J. (2024). Color Equivariant Convolutional Networks. Advances in Neural Information Processing Systems, 36.

<a id="1">[color_selectivity]</a>
Ivet Rafegas and Maria Vanrell. Color encoding in biologically-inspired convolutional neural  networks. Vision Research, 151:7–17, 2018. Color: cone opponency and beyond.



