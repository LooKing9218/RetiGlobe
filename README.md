# RetiGlobe

Foundation models show great potential for advancing imaging diagnostics; however, their reliance on fine-tuning for downstream tasks, particularly when applied across different clinical settings, limits their scalability and deployment. We introduce GlobeReady, a clinician-friendly AI platform that enables ocular disease diagnosis that operates without the need for fine-tuning, or specialized technical expertise. The platform is powered by RetiGlobe, a foundation model specifically engineered for ophthalmic image understanding. To pre-train RetiGlobe, we first generated a large-scale synthetic dataset of 38 million ophthalmic images using a generative model. We then curated 475,845 real-world image-text pairs from globally diverse sources. We initially employed the self-supervised DINOv2 framework on the synthetic dataset to enable the model to learn detailed structural and pathological features. Next, we conducted contrastive pretraining with the CLIP framework on the real image-text pairs, aligning visual features with semantic context to enhance complex feature understanding. These two complementary pretraining stages endow RetiGlobe with robust and generalizable ophthalmic knowledge. Building upon RetiGlobe’s capabilities, we introduced the innovative GlobeReady platform, a completely code-free and training-free tool. Clinicians can effortlessly upload local data to create customized feature repositories, performing tasks such as disease diagnosis via simple feature matching. This streamlined workflow significantly simplifies the integration and practical deployment of AI technologies into everyday clinical practice.
# Demo video of our GlobeReady platform can be accessed by the following link:
https://looking9218.github.io/GlobeReady/globe_ready_demo.html

![PretrainingOverview](https://github.com/user-attachments/assets/af303852-436a-4644-89a3-fc22bb8cc975)

# Pre-trained weights for RetiGlobe:

![Pretrained weights for CFP] (https://drive.google.com/file/d/1TBfUppEqA9i3WMIz62xgwKWtN2JDBZyy)

![Pretrained weights for OCT] (https://drive.google.com/file/d/1agoH70WM9ulyym6uKX3yrEk_4FXfxPvX)
