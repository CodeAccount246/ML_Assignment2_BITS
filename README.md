# Netflix Content Classification & Evaluation App
## a. Problem Statement
The objective of this project is to build, evaluate, and benchmark five distinct machine learning classification algorithms to predict whether a Netflix title is a Movie (0) or a TV Show (1).

## c. GitHub Repository Link
`https://github.com/2025NS04135/ML-Assignment-2-BITS`

## f. Streamlit Deployment & Secrets Management

To deploy this application to Streamlit Community Cloud:

1. **GitHub Setup:** Push this repository to GitHub. Ensure you include `app.py`, `requirements.txt`, `test_data.csv`, and the `model/` folder.
2. **Ignore Secrets:** Ensure your `.streamlit/secrets.toml` is added to your `.gitignore` file so it is not pushed to your public repository.
3. **Streamlit Cloud:** Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
4. **Deploy:** Click "New app", select your repository, branch, and specify `app.py` as the main file path.
5. **Manage Secrets in Cloud:** Before clicking Deploy (or in the App Settings > Secrets after deploying), paste the exact contents of your local `secrets.toml` into the text box. Streamlit will securely inject these into the `st.secrets` dictionary at runtime.
