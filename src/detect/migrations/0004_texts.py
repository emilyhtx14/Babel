# Generated by Django 3.1.6 on 2021-02-14 02:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0003_record'),
    ]

    operations = [
        migrations.CreateModel(
            name='Texts',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('passage', models.CharField(max_length=1000)),
            ],
        ),
    ]
