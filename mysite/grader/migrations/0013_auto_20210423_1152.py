# Generated by Django 3.1.7 on 2021-04-23 11:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('grader', '0012_auto_20210422_2147'),
    ]

    operations = [
        migrations.AlterField(
            model_name='essay',
            name='image',
            field=models.ImageField(null=True, upload_to='images'),
        ),
    ]
