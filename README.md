## Campaign Perfomance and Planner


### Introduction

In today’s digital era, marketing has considerably evolved in data availability and technology. From ad hoc reporting insights to AI-powered machine learning methods, data driven decisions have become an integral part of marketing in various industries. Two popular themes include campaign effectiveness and forecasting. Marketing teams are spearheading the use of machine learning with their campaigns to engage with their customers. Fine tuning the campaign process by targeting groups of individuals who are more likely to install or subscribe to a product allows for companies to allocate their marketing resources and budget more efficiently.

This project aims to satisfy marketers’ needs in two ways: 
1. <b>Campaign Effectiveness:</b> A one-stop shop tool for marketers to be able analyze their campaigns visually and gain high level insights.
2. <b>Campaign Planning/Forecasting:</b> Plan future campaigns using various machine and deep learning techniques such as random rainforest and gradient boost regressors.

A beta version of the dashboard can be viewed <b>[HERE](https://campaign-manager.streamlit.app/)</b>.


### Business Objectives

Below are few of the many business questions we're looking to answer.
- What is the impact of campaign spend from this week to next week for total active users, viewed minutes, and minutes per user?
- For campaign budgeting optimization for upcoming quarters, how can marketing best spend their budget to optimize spend across marketing partners to increase ROI?
- Which states bring in the most users through campaign ads? Which states have the most total minutes watched per user from an ad?


## Motivation

Marketing research covers a spectrum from applied methods like A/B testing to theoretical discussions on marketing’s strategic value. Marketing Performance Measurement (MPM) enhances a firm’s capacity to evaluate how marketing efforts drive business outcomes (O’Sullivan and Abela, 2007). Strong MPM systems are essential for justifying marketing expenses and building credibility within the organization. Firms with robust MPM capabilities show superior financial results, such as higher profitability and sales growth.

According to Clemens Koob's research Determinants of content marketing effectiveness, the factors contributing to the success of content marketing campaigns are explored, along with strategies to increase their effectiveness. One key finding is that neither the number of media platforms used nor the specific distribution channels significantly influence campaign outcomes. Instead, the study emphasizes that the quality and alignment of content with the audience's interests are the primary drivers of success.

It's also worth noting that aligning content with audience needs is essential for success (Lopes and Casais, 2022). The strategic framework can categorize companies as “emerging,” “developing,” or “maturing” based on their content marketing practices.


## Getting Started

### Dependencies

- Prerequisites: Requires a programming tool of choice (Jupyter Notebook, VS Code, Google Colab, etc)
- Libraries: See [requirements.txt](https://github.com/nazcat/campaign_manager_tool/blob/main/requirements.txt)

### Installing

File downloads: Refer to our GDrive to download a total of four files from [Streamlit Datasets](https://drive.google.com/drive/folders/1_Tq1ZCAZNYtc6vUbpKIpBLeoZeKRvH38?usp=sharing):
  - [anon_agg_for_models.csv](https://drive.google.com/file/d/1PZzmOjY8bl-ZSRmrPxTx1pmU6sXmuV3x/view?usp=sharing) [BASE DATASET] - Aggregate datafile sampled to 125,000 records with encrypted features (i.e. device_id, campaign_id, etc)
  - [anan_campaign_modeling_data_v3.csv](https://drive.google.com/file/d/1RQabm5Sh0MtiJoi1zKyHGJc049y09JBB/view?usp=sharing) - Data aggregated to the campaign level
  - [total_genre.csv](https://drive.google.com/file/d/19PRykaEUS-lHebwqvpzBe4v5OGNmAUmh/view?usp=sharing) - Data aggregated to the genre level
  - [totals.csv](https://drive.google.com/file/d/1PHRAEzjbOcjqLb3I4aZbHwM2-iezdo0G/view?usp=sharing) - Data aggregated to the marketing partner level

NOTE: For access to full base dataset (8.6M records instead of 125k), in place of [anon_agg_for_models.csv](https://drive.google.com/file/d/1PZzmOjY8bl-ZSRmrPxTx1pmU6sXmuV3x/view?usp=sharing), refer to [anon_processed_unique_device_v3.csv](https://drive.google.com/file/d/1_HQZWSKlwm8b_S4XZQccwrW-uQF9YxMk/view?usp=sharing).

The code is accessible in two different ways:
1. Using either command prompt for Windows or terminal for mac, clone Github repository:
```
git clone https://github.com/nazcat/campaign_manager_tool.git
```
2. Download .zip file from Github in the [campaign_manager_tool](https://github.com/nazcat/campaign_manager_tool/) repository, located under the green "code" dropdown at the top.


### Executing program

In order to run the Streamlit app locally, follow the below in order:
1. Open command prompt or terminal, and open the repository location:
```
cd file_path/
```
2. Open locally in Streamlit:
```
streamlit run campaign_manager_visuals_only.py
```

## Closing Thoughts

[ADD TEXT HERE]


## Authors

Contributors names and contact info

Naz Uremek
Calvin Raab
Chris Struck
Justin Barry

## Version History

* 0.1
    * Initial Release


## References

- Koob, Clemens. “Determinants of Content Marketing Effectiveness: Conceptual Framework and empirical findings from a managerial perspective.” PLOS ONE, vol. 16, no. 4, 1 Apr. 2021, https://doi.org/10.1371/journal.pone.0249457. 
- Lopes, Ana, and Beatriz Casais. “Digital Content Marketing: Conceptual Review and Recommendations for Practitioners.” Academy of Strategic Management Journal, vol. 21,  Jan. 2022, pp. 1-17, 
https://www.researchgate.net/publication/357746605_Digital_Content_Marketing_Conceptual_Review_and_Recommendations_for_Practitioners.
- O’Sullivan, Don, and Andrew V Abela. “Marketing performance measurement ability and firm performance.” Journal of Marketing, vol. 71, no. 2, Apr. 2007, pp. 79–93, https://doi.org/10.1509/jmkg.71.2.79. 
- Sarkar, Mainak, and Arnaud De Bruyn. “LSTM response models for direct marketing analytics: Replacing feature engineering with Deep Learning.” Journal of Interactive Marketing, 2020, https://doi.org/10.2139/ssrn.3601025. 
