use opencv::{
    core::Mat,
    highgui::{self},
    imgcodecs, imgproc,
};
use rfd::FileDialog;
use std::cmp::max;

fn main() -> anyhow::Result<()> {
    println!("opencv version: {}", opencv::core::CV_VERSION);

    let binding = FileDialog::new()
        .add_filter("image", &["jpg", "png", "jpeg", "bmp", "tiff"])
        .set_directory("/")
        .pick_file()
        .expect("File selection failed");

    let path = binding.to_str(); // "D:\\projects\\rust\\getStarted\\start\\img\\cells.jpg";
    highgui::named_window("window", 1)?;

    let mut ced_param: f64;
    let mut hough_param: f64;
    let mut hough_param_int = 120;
    let mut ced_param_int = 200;

    highgui::create_trackbar(
        "hough_param",
        "window",
        Some(&mut hough_param_int),
        400,
        None,
    )?;
    highgui::create_trackbar("canny_param", "window", Some(&mut ced_param_int), 400, None)?;
    // Open the web-camera (assuming you have one)
    loop {
        hough_param = max(1, hough_param_int / 10) as f64;
        ced_param = max(ced_param_int, 1) as f64;
        let mut img = imgcodecs::imread(path.expect("Invalid filepath"), imgcodecs::IMREAD_COLOR)?;

        let mut gray: opencv::core::Mat = Default::default();
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut thresh: opencv::core::Mat = Default::default();
        imgproc::threshold(
            &gray,
            &mut thresh,
            0.0,
            255.0,
            imgproc::THRESH_BINARY_INV + imgproc::THRESH_OTSU,
        )?;

        //Circles
        let mut canny = Mat::default();
        imgproc::canny_def(&gray, &mut canny, ced_param, ced_param / 2.0)?;
        let mut circles: opencv::core::Vector<opencv::core::Vec3f> = Default::default();
        imgproc::hough_circles(
            &gray,
            &mut circles,
            imgproc::HOUGH_GRADIENT,
            1.0,
            4.0,
            ced_param,
            hough_param,
            0,
            20,
        )?;

        println!("{}", circles.len());
        for a in circles {
            let x = a.get(0).expect("Invalid circle x");
            let y = a.get(1).expect("Invalid circle x");
            let r = a.get(2).expect("Invalid circle x");
            imgproc::circle_def(
                &mut img,
                opencv::core::Point::new(*x as i32, *y as i32),
                *r as i32,
                opencv::core::Scalar::new(255.0, 0.0, 255.0, 1.0),
            )?;
            // and display in the window
        }
        highgui::imshow("gray", &gray)?;
        highgui::imshow("window", &canny)?;
        highgui::imshow("original", &img)?;
        let key = highgui::wait_key(1)?;
        if key == 113 {
            // quit with q
            break;
        }
    }
    Ok(())
}
