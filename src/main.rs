use opencv::{
    core::Mat,
    highgui::{self},
    imgcodecs, imgproc,
};
use std::cmp::max;

fn main() -> anyhow::Result<()> {
    println!("opencv version: {}", opencv::core::CV_VERSION);
    let path = "D:\\projects\\rust\\getStarted\\start\\img\\cells.jpg";
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
        let mut img = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;

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
        for index in 0..(circles.len() - 1) {
            let a = circles.get(index).unwrap();
            let x = a.get(0).unwrap();
            let y = a.get(1).unwrap();
            let r = a.get(2).unwrap();
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
