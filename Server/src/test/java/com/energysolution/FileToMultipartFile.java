package com.energysolution;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.disk.DiskFileItem;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.commons.CommonsMultipartFile;


public class FileToMultipartFile {
	@Test
	void test() {
//		FileItem fileItem = new DiskFileItem("file", File.probeContentType(file.toPath()), false, file.getName(), (int) file.length() , file.getParentFile());
//		InputStream input = new FileInputStream(file);
//		OutputStream os = fileItem.getOutputStream();
//		IOUtils.copy(input, os);
//		MultipartFile multipartFile = new CommonsMultipartFile(fileItem);
	}
}
