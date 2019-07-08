---
title: "MNYs Deep Learning Self Driving Car"
header:
  teaser: /assets/images/AD.png
categories:
  - Self Driving Car
  - Deep Learning
  - Neural Network
  - Maximilian Christian Roth
  - Nina Claire Pant
  - Yvan Putra Satyawan
tags:
  - content
  - css
  - markup
---

![Hero Image](/assets/simulator-1.png)

### Some Context
This is our bachelor project at the Chair for Autonomous Intelligent Systems (AIS) at the University of Freiburg.<br>

It was on a self-driving car based on the paper "End-to-End Driving via Conditional Imitation Learning" by Codevilla et al.<br>

Here we implement the architecture given in the paper and attempted to create variations and improvements,<br>

including a version which first attempts to first perform semantic segmentation on the camera image first.<br>

### Video Demo
<div class="embed-container">
  <iframe
      src="https://player.vimeo.com/video/346883000"
      width="640"
      height="564"
      frameborder="0"
      webkitallowfullscreen
      mozallowfullscreen
      allowfullscreen>
  </iframe>
</div>

### Variations & Improvements
We made several different variations to the original paper.<br>
The first was to use only one camera instead of the proposed three.<br>
This was a necessity, due to hardware limitations of the tools provided.<br>
We later managed to expand this to two cameras by rewriting some of the code written by the supervisors.<br><br>

<!--
   -  | Network           | Description
   -  |-------------------|-------------
   -  | standard          | Uses only one forward facing camera to drive
   -  | segmented         | Uses only the ground truth segmentation of a forward facing camera
   -  | seg and normal    | Uses both the image and ground truth segmentation of a forward facing camera
   -  | last image too    | Uses both the current and previous image from a forward facing camera
   -  | two cams          | Uses a forward facing and a camera angled towards the right
   -  | self segmentation | Uses the forward facing camera and segments the image with a network first
   -
   -<style type="text/css">
   -.tg  {border-collapse:collapse;border-spacing:0;border-color:#9ABAD9;}
   -.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#9ABAD9;color:#444;background-color:#EBF5FF;}
   -.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#9ABAD9;color:#fff;background-color:#409cff;}
   -.tg .tg-1wig{font-weight:bold;text-align:left;vertical-align:top}
   -.tg .tg-hmp3{background-color:#D2E4FC;text-align:left;vertical-align:top}
   -.tg .tg-sf3y{background-color:#D2E4FC;font-family:Arial, Helvetica, sans-serif !important;;text-align:left;vertical-align:top}
   -.tg .tg-0lax{text-align:left;vertical-align:top}
   -</style>
   -<table class="tg">
   -  <tr>
   -    <th class="tg-1wig">Network</th>
   -    <th class="tg-1wig">Description</th>
   -  </tr>
   -  <tr>
   -    <td class="tg-hmp3">standard</td>
   -    <td class="tg-hmp3">Uses only one forward facing camera to drive</td>
   -  </tr>
   -  <tr>
   -    <td class="tg-0lax">segmented</td>
   -    <td class="tg-0lax">Uses only the ground truth segmentation of a forward facing camera</td>
   -  </tr>
   -  <tr>
   -    <td class="tg-hmp3">seg and normal</td>
   -    <td class="tg-sf3y">Uses both the image and ground truth segmentation of a forward facing camera</td>
   -  </tr>
   -  <tr>
   -    <td class="tg-0lax">last image too</td>
   -    <td class="tg-0lax">Uses both the current and previous image from a forward facing camera</td>
   -  </tr>
   -  <tr>
   -    <td class="tg-hmp3">two cams</td>
   -    <td class="tg-hmp3">Uses a forward facing and a camera angled towards the right</td>
   -  </tr>
   -  <tr>
   -    <td class="tg-0lax">self segmentation</td>
   -    <td class="tg-0lax">Uses the forward facing camera and segments the image with a network first</td>
   -  </tr>
   -</table>
   -->
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 15px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-tebh{background-color:#f9f9f9;font-family:Arial, Helvetica, sans-serif !important;;text-align:left;vertical-align:top}
.tg .tg-buh4{background-color:#f9f9f9;text-align:left;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-amwm">Network</th>
    <th class="tg-amwm">Description</th>
  </tr>
  <tr>
    <td class="tg-buh4">standard</td>
    <td class="tg-buh4">Uses only one forward facing camera to drive</td>
  </tr>
  <tr>
    <td class="tg-0lax">segmented</td>
    <td class="tg-0lax">Uses only the ground truth segmentation of a forward facing camera</td>
  </tr>
  <tr>
    <td class="tg-buh4">seg and normal</td>
    <td class="tg-tebh">Uses both the image and ground truth segmentation of a forward facing camera</td>
  </tr>
  <tr>
    <td class="tg-0lax">last image too</td>
    <td class="tg-0lax">Uses both the current and previous image from a forward facing camera</td>
  </tr>
  <tr>
    <td class="tg-buh4">two cams</td>
    <td class="tg-buh4">Uses a forward facing and a camera angled towards the right</td>
  </tr>
  <tr>
    <td class="tg-0lax">self segmentation</td>
    <td class="tg-0lax">Uses the forward facing camera and segments the image with a network first</td>
  </tr>
</table>

<!--Infos on the different variations-->
<details> 
  <summary>Standard</summary>
   <p>  Single camera and otherwise the implementation from the paper.<br>
        Works pretty well and is shown on the video.<br>
        This is on the testing dataset / unknown map and only with a small amount of data.</p> 
</details>
<details> 
  <summary>Segmented</summary>
   <p> Has the ground truth segmentation of a single camera output.<br>
        As expected the results of this artificial neural network are extremely
        good and it handles even difficult problems with ease.<br>
        Obtaining such a perfect segmentation in reality is of course very hard if not impossible.</p> 
        Had some problems with lanes coming from the opposite direction, as they had the same coloring<br>
        (eg. left lane one color right lane another) and the left lanes from both sides had the same one.
</details>
<details> 
  <summary>Seg And Normal</summary>
   <p> Had both the original image of a camera as well as its ground truth segmentation.<br>
        It did not perform better in most situations, as it carried with it some errors of the image only,<br>
        like thinking multiple parking spaces in a row are another lane.<br>
        It performed better than the segmentation only on streets with two lanes for each direction,<br>
        this was likely becuase it now saw the line separating the two directions.
        In general it was perfoming on the same level as seg. only.</p> 
</details>
<details> 
  <summary>Last Image Too</summary>
   <p> This one was a bit tough and brought some problems,<br>
        as we doubled the input space it saved something similar to a state machine.<br>
        It tended to overcorrect, as it thought it was still very bad,<br>
        when it saw the last picture as the same as the new one before.<br>
        Though this was probably only because we trained it with the normal dataset.<br>
        In the end we were constrained by the end of the project.</p> 
</details>
<details> 
  <summary>Two Cams</summary>
   <p> A second image from the same position facing a bit to the left,<br>
        though in hindsight we should've done it to the right,<br>
        to get the outer side of the lanes in better view.<br>
        This overcorrected to the left, as it had the same problem as last img. too.<br>
        With a bigger dataset both would've probably just ignored the other image.</p> 
</details>
<details> 
  <summary>Self Segmentation</summary>
   <p> Here we implemented our own segmentation network to segment street,<br>
        lines indicating lanes and unimportant stuff.<br>
        Sadly on the computers provided this could barely run in real time and<br>
        lagged terribly, even though it worked quite well, if slow on our GPU.</p> 
</details>

For more info on the structure of the neural networks look into our [slides](/assets/AIS-Project.pdf)

<style>
.responsive-wrap iframe{ max-width: 100%;}
</style>
<div class="responsive-wrap">
<!-- this is the embed code provided by Google -->
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTzGNXiUU3pggNZtoAv8Kh121AKM4Yoeise1na-jsMc2sZhS_MFcNcQCBsZy0yDV1Sl_UCXugQpBded/embed?start=false&loop=true&delayms=5000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
<!-- Google embed ends -->
</div>

### Environment
The whole simulation took place on a test map similar to the [**Audi Cup**]("https://www.audicup.com/" "Audi Cup") inside [**Unreal Engine**]("https://www.audicup.com/" "Unreal Engine").<br>
