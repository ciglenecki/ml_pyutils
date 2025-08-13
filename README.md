# ml_pyutils

Summary: collection of useful Python functions and modules which I commonly use in the context of machine learning workflows

Design decisions:
- single module dependecy: each module should not depend on other models inside of the repository
- each utility function should be self contained
  
## generate fake video

```
ffmpeg -f lavfi -i testsrc=duration=5:size=300x300:rate=30   -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:        text='%{n}':        fontcolor=white:        fontsize=h/12:        x=(w-text_w)/2:        y=(h-text_h)/2:        box=1: boxcolor=black@0.5"   -c:v mpeg4 -qscale:v 1   -y centered_numbered.avi
```
