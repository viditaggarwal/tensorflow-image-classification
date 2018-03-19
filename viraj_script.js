var images = [];
$.each($(".s-item img"), function(index, img) {
    if(img.src.indexOf(".jpg") !==-1 || img.src.indexOf(".JPG") !==-1|| img.src.indexOf(".jpeg") !==-1 || img.src.indexOf(".webp") !==-1 || img.src.indexOf(".png") !==-1)
      images.push(img.src)
})
console.log(images);