name: Auto PR and Comment

on:
  schedule:
    - cron: "7 5 * * *"  # 매일 오후 2시 3분 실행 (KTC 기준) UTC 5시 3분
    
jobs:
  create-pr-and-comment:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    # PR 생성
    - name: Create Pull Request
      id: create_pr
      uses: peter-evans/create-pull-request@v4
      with:
        commit-message: "Auto update for testing"
        title: "Auto-generated PR for testing"
        body: "This PR is auto-generated for testing purposes."
        base: main  # PR의 타겟 브랜치
        branch: auto-generated-branch  # PR의 새로운 브랜치 이름
        delete-branch: true  # PR이 머지되면 자동으로 브랜치를 삭제

    # PR에 코멘트 추가
    - name: Add comment to PR
      uses: actions/github-script@v6
      with:
        script: |
          const pr_number = ${{ steps.create_pr.outputs.pull_request_number }};
          github.rest.issues.createComment({
            issue_number: pr_number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: '/test'
          });
