---
categories:
- linux
date: "2021-02-11T00:00:00Z"
tags: null
title: docker 로 gitlab만들기
---

# 도커 이미지 다운 및 실행 

```
docker run --detach \
 --hostname gitlab.example.com \
 --publish 443:443 --publish 80:80 --publish 22:22 \
 --name gitlab \
 --restart always \
 --volume /srv/gitlab/config:/etc/gitlab \
 --volume /srv/gitlab/logs:/var/log/gitlab \
 --volume /srv/gitlab/data:/var/opt/gitlab \
 gitlab/gitlab-ce:latest
 ```

ip가 있을시에 gitlab.example.com 가 hostname이 된다. http접속을 위한 80번 포트 https 를 위한 443포트 ssh 접속을 위한 22번 포트를 열어준다. 데이터 백업을 위하여 config,logs,data에 볼륨을 백업한다

localhost로 접속하여 설정을 진행한다

# SMTP 설정(GMail기준)
gmail의 smtp 설정을 한후 아래 명령어로 gitlab.rb에 들어간후 수정한다.

```
# Email Settings
gitlab_rails['smtp_enable'] = true
gitlab_rails['gitlab_email_enabled'] = true
gitlab_rails['smtp_address'] = "smtp.gmail.com"
gitlab_rails['smtp_port'] = 587
gitlab_rails['smtp_user_name'] = "username@gmail.com"
gitlab_rails['smtp_password'] = "password"
gitlab_rails['smtp_domain'] = "smtp.gmail.com"
gitlab_rails['smtp_authentication'] = "login"
gitlab_rails['smtp_enable_starttls_auto'] = true
gitlab_rails['smtp_tls'] = false
gitlab_rails['smtp_openssl_verify_mode'] = 'peer'
```
# HTTPS 적용

## letsencrypt 설치
```
sudo apt-get install letsencrypt
```
## docker exec -it gitlab vim /etc/gitlab/gitlab.rb 에서 아래구문 추가
```
nginx['custom_gitlab_server_config'] = "location ^~ /.well-known { root /var/www/letsencrypt; }"
```
## 설정 적용
```
docker exec -it gitlab gitlab-ctl reconfigure
```
## 인증서 발급
```
sudo letsencrypt certonly -a webroot -w /var/www/letsencrypt -d gitlab.example.com
```
## docker exec -it gitlab vim /etc/gitlab/gitlab.rb 에서 아래구문 추가
```
nginx['redirect_http_to_https']=true
nginx['ssl_certificate'] = "/etc/letsencrypt/live/#{node['fqdn']}/fullchain.pem"
nginx['ssl_certificate_key'] = "/etc/letsencrypt/live/#{node['fqdn']}/privkey.pem"
```
## 설정 적용
```
docker exec -it gitlab gitlab-ctl reconfigure
```
## SSL 인증서 자동갱신 설정
```
crontab -e
10 5 * * 1 /usr/bin/letsencrypt renew >> /var/log/le-renew.log
15 5 * * 1 /usr/bin/gitlab-ctl restart nginx
```
# references
- https://lovemewithoutall.github.io/it/start-docker/
- https://blog.lael.be/post/5476
