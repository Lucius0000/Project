create database yolo_app;

use yolo_app;

create table users (
id int auto_increment primary key,
username varchar(100) not null unique,
password varchar(255) not null
);