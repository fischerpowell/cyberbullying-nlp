import instaloader
from password import password
import csv
import os
import cleantext

ig = instaloader.Instaloader()

ig.login('USERNAME123', password)

hashtags = ['classicmenswear']

max_comments = 5

max_posts = 5

with open('temp.csv', 'w') as csvfile:
    fieldnames = ['hashtag', 'op', 'comment', 'timestamp']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='', lineterminator='\n')
    writer.writeheader()
    comment_count = 0
    post_count = 0
    for hashtag in hashtags:
        for post in instaloader.Hashtag.from_name(ig.context, hashtag).get_posts():
            op = post.profile
            wrote_comment = False
            for comment in post.get_comments():
                comment_text = cleantext.clean(comment.text, no_emoji=True)
                if comment_text != '':
                    writer.writerow({'hashtag' : hashtag,
                                    'op' : op,
                                    'comment' : comment_text,
                                    'timestamp' : comment.created_at_utc
                                    })
                    comment_count += 1
                    wrote_comment = True
                
                if comment_count == max_comments:
                    break
            if wrote_comment:
                post_count += 1
            if post_count == max_posts:
                break
    csvfile.close()