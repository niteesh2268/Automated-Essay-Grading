# Generated by Django 3.1.7 on 2021-04-22 21:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('grader', '0010_auto_20210422_2141'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='essay',
            name='min_score',
        ),
        migrations.AlterField(
            model_name='essay',
            name='image',
            field=models.ImageField(null=True, upload_to=''),
        ),
    ]
