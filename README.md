## Campaign Performance and Predictions Tool

### Introduction

In today’s digital era, marketing has considerably evolved in data availability and technology. From ad hoc reporting insights to AI-powered machine learning methods, data driven decisions have become an integral part of marketing in various industries. Two popular themes include campaign effectiveness and forecasting. Marketing teams are spearheading the use of machine learning with their campaigns to engage with their customers. Fine tuning the campaign process by targeting groups of individuals who are more likely to install or subscribe to a product allows for companies to allocate their marketing resources and budget more efficiently.

This project aims to satisfy marketers’ needs in two ways: 
1. Campaign Effectiveness: A one-stop shop tool for marketers to be able analyze their campaigns visually and gain high level insights.
2. Campaign Planning/Forecasting: Plan future campaigns using various machine and deep learning techniques such as random rainforest and gradient boost regressors.

A beta version of the dashboard can be viewed <b>[HERE](https://campaign-manager-visuals.streamlit.app/)</b>.

### Business Objectives

Below are few of the many business questions we're looking to answer.
- What is the impact of campaign spend from this week to next week for total active users, viewed minutes, and minutes per user?
- For campaign budgeting optimization for upcoming quarters, how can marketing best spend their budget to optimize spend across marketing partners to increase ROI?
- Which states bring in the most users through campaign ads? Which states have the most total minutes watched per user from an ad?

## Getting Started

### Dependencies

- Prerequisites: Requires a programming tool of choice (Jupyter Notebook, VS Code, Google Colab, etc)
- Libraries: See [requirements.txt](https://github.com/nazcat/campaign_manager_tool/blob/main/requirements.txt)

### Installing

File downloads: Refer to our GDrive to download a total of four files from [Streamlit Datasets](https://drive.google.com/drive/folders/1_Tq1ZCAZNYtc6vUbpKIpBLeoZeKRvH38?usp=sharing):
  - [anon_processed_unique_device_v3.csv](https://drive.google.com/file/d/1PvHjL6HEmSuEix0vLT3FJalf7NEm15Ql/view?usp=sharing) -  Raw datafile with encrypted features (i.e. device_id, campaign_id, etc)
  - [anan_campaign_modeling_data_v3.csv](https://drive.google.com/file/d/1RQabm5Sh0MtiJoi1zKyHGJc049y09JBB/view?usp=sharing) - Data aggregated to the campaign level
  - [total_genre.csv](https://drive.google.com/file/d/19PRykaEUS-lHebwqvpzBe4v5OGNmAUmh/view?usp=sharing) - Data aggregated to the genre level
  - [totals.csv](https://drive.google.com/file/d/1PHRAEzjbOcjqLb3I4aZbHwM2-iezdo0G/view?usp=sharing) - Data aggregated to the marketing partner level

The code is accessible in two different ways:
1. Using either command prompt for Windows or terminal for mac, clone Github repository:
```
git clone https://github.com/nazcat/campaign_manager_tool.git
```
2. Download .zip file from Github in the [campaign_manager_tool](https://github.com/nazcat/campaign_manager_tool/tree/main) repository, located under the green "code" dropdown at the top.


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

NOTE FOR BOTH OPTIONS: Due to file size limitations of 25MB in Github, you will need to download [anon_processed_unique_device_v3.csv](https://drive.google.com/file/d/1PvHjL6HEmSuEix0vLT3FJalf7NEm15Ql/view?usp=sharing) separately from GDrive. Make sure to add to the /data/ folder.


## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
