# Generated by Django 3.1.6 on 2021-02-14 00:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0002_auto_20210213_1453'),
    ]

    operations = [
        migrations.CreateModel(
            name='Record',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('recording', models.CharField(max_length=120)),
            ],
        ),
    ]
