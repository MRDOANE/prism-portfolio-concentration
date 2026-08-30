# GitHub and Zenodo release guide

This guide publishes the repository as
`https://github.com/MRDOANE/prism-portfolio-concentration` and archives release
`v1.0.0` in Zenodo.

## 1. Extract and inspect the release ZIP

1. Download the GitHub-ready ZIP.
2. In Windows Explorer, right-click the ZIP and select **Extract All**.
3. Open the extracted `prism-portfolio-concentration` folder.
4. Confirm that `README.md`, `CITATION.cff`, `LICENSE`, `config`, `prism_sim`,
   `scripts`, `tests`, `results_e4_e7`, and `results_e8` are present.

## 2. Create the local repository in GitHub Desktop

1. Open GitHub Desktop and sign in to the `MRDOANE` GitHub account.
2. Select **File > Add local repository**.
3. Choose the extracted `prism-portfolio-concentration` folder.
4. If GitHub Desktop says the folder is not a Git repository, click **create a
   repository** when prompted.
5. Use:
   - Name: `prism-portfolio-concentration`
   - Description: `Simulation code and reference results for risk capacity,
     synergy, and concentration in pharmaceutical R&D portfolios.`
   - Git ignore: **None**
   - License: **None**
6. Create the repository in the existing extracted folder. The ZIP already
   contains `.gitignore` and `LICENSE`.
7. In the Changes panel, confirm that the scientific files are present and that
   generated folders such as `.venv` are absent.
8. Enter the commit summary `Initial public release v1.0.0` and click **Commit
   to main**.
9. Click **Publish repository**.
10. Keep the name `prism-portfolio-concentration`, select the personal account,
    and clear **Keep this code private**.
11. Click **Publish Repository**.

## 3. Finish the GitHub landing page

On the repository page in GitHub:

1. Add these topics:
   - `pharmaceutical-rd`
   - `portfolio-management`
   - `decision-analysis`
   - `monte-carlo-simulation`
   - `risk-management`
   - `enpv`
   - `python`
2. Confirm that the README renders and that the Actions tab shows the `tests`
   workflow.
3. Wait for the workflow to finish with a green check.

## 4. Connect the repository to Zenodo before releasing

1. Sign in to Zenodo.
2. Open the profile menu and select **GitHub**.
3. If GitHub is not connected, connect and authorize the `MRDOANE` account.
4. Click **Sync now**.
5. Find `MRDOANE/prism-portfolio-concentration`.
6. Turn on the repository's Zenodo toggle.
7. Refresh the page and confirm that the repository remains enabled.

Do this before creating the GitHub release. Zenodo archives releases created
after the repository has been enabled.

## 5. Create GitHub release v1.0.0

1. Return to the GitHub repository.
2. Open **Releases** and select **Draft a new release**.
3. Select **Choose a tag**.
4. Enter `v1.0.0` and choose **Create new tag: v1.0.0 on publish**.
5. Target the `main` branch.
6. Release title: `v1.0.0 - Beyond Independent Shots on Goal`.
7. Paste the contents of `RELEASE_NOTES_v1.0.0.md` into the description.
8. Leave **Set as a pre-release** cleared.
9. Select **Set as the latest release**.
10. Click **Publish release**.

The GitHub-generated source archive is sufficient. The original setup ZIP does
not need to be attached to the release.

## 6. Verify and publish the Zenodo record

1. Return to **Zenodo > GitHub** and open the enabled repository.
2. Wait for `v1.0.0` to finish processing.
3. Open the resulting Zenodo record and verify:
   - Resource type: **Software**
   - Title: **PRISM Portfolio Concentration Simulation**
   - Creator: **Doane, Michael R.**
   - ORCID: **0009-0003-0521-8981**
   - Version: **1.0.0**
   - Access: **Open**
   - License: **MIT**
4. Confirm that the archived files correspond to GitHub tag `v1.0.0`.
5. Record both identifiers displayed by Zenodo:
   - the version-specific DOI for `v1.0.0`;
   - the concept DOI that resolves to the latest version.

Use the version-specific DOI in the manuscript's Data Availability statement.
The concept DOI is appropriate for a README badge intended to follow future
releases.

## 7. Add the DOI back to GitHub and the manuscript

After Zenodo assigns the DOI:

1. On the Zenodo record, copy the Markdown DOI badge.
2. Edit `README.md` on GitHub and place the badge below the existing test badge.
3. Edit `CITATION.cff` and add:

   ```yaml
   doi: "10.5281/zenodo.RECORD_NUMBER"
   ```

4. Commit those two changes with the message `Add Zenodo DOI`.
5. Update the manuscript Data Availability statement with the GitHub URL and
   version-specific DOI.

There is no need to create `v1.0.1` solely to add the DOI badge. Create a new
release when the scientific code or reference results change.

## Troubleshooting

- If the repository does not appear in Zenodo, click **Sync now** and check that
  Zenodo is authorized for the `MRDOANE` account.
- If a release fails, open that release under **Zenodo > GitHub** and expand
  **Errors**. Metadata parsing is the first item to check.
- This repository uses `CITATION.cff` and intentionally omits `.zenodo.json` so
  Zenodo has one metadata source.
- If GitHub Actions fails while installing dependencies, open the failed job and
  save the log before changing version pins.

