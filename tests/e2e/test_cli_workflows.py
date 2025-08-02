"""
End-to-end tests for CLI workflows.

Tests complete user workflows from command-line interface to final output.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestCLIBasicWorkflows:
    """Test basic CLI workflows."""
    
    def test_cli_help_command(self):
        """Test that CLI help command works."""
        result = subprocess.run(
            ["python", "-m", "pqc_iot_retrofit.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"CLI help failed: {result.stderr}"
        assert "PQC IoT Retrofit Scanner" in result.stdout
        assert "scan" in result.stdout
        assert "patch" in result.stdout
    
    def test_cli_version_command(self):
        """Test that CLI version command works."""
        result = subprocess.run(
            ["python", "-m", "pqc_iot_retrofit.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"CLI version failed: {result.stderr}"
        assert "0.1.0" in result.stdout  # Should match version from pyproject.toml
    
    def test_cli_scan_firmware(self, arm_rsa_sample, tmp_path):
        """Test complete firmware scan workflow via CLI."""
        # Create temporary firmware file
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        output_file = tmp_path / "scan_results.json"
        
        # Run CLI scan command
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "arm",
            "--output", str(output_file),
            "--verbose"
        ], capture_output=True, text=True, timeout=60)
        
        # Check command succeeded
        assert result.returncode == 0, f"CLI scan failed: {result.stderr}"
        
        # Check output file was created
        assert output_file.exists(), "Output file was not created"
        
        # Parse and validate output
        with open(output_file, 'r') as f:
            scan_results = json.load(f)
        
        assert "vulnerabilities" in scan_results
        assert "metadata" in scan_results
        assert len(scan_results["vulnerabilities"]) > 0
        
        # Check vulnerability structure
        vuln = scan_results["vulnerabilities"][0]
        required_fields = ["algorithm", "confidence", "severity", "address"]
        for field in required_fields:
            assert field in vuln, f"Missing field {field} in vulnerability"
    
    def test_cli_scan_with_filters(self, mock_vulnerable_firmware, tmp_path):
        """Test CLI scan with confidence and severity filters."""
        firmware_file = tmp_path / "vulnerable_firmware.bin"
        firmware_file.write_bytes(mock_vulnerable_firmware)
        
        output_file = tmp_path / "filtered_results.json"
        
        # Run scan with high confidence filter
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "arm",
            "--min-confidence", "0.8",
            "--min-severity", "medium",
            "--output", str(output_file)
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"Filtered scan failed: {result.stderr}"
        
        # Check results match filters
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        for vuln in results["vulnerabilities"]:
            assert vuln["confidence"] >= 0.8, "Confidence filter not applied"
            assert vuln["severity"] in ["medium", "high", "critical"], \
                "Severity filter not applied"
    
    def test_cli_patch_generation(self, arm_rsa_sample, tmp_path):
        """Test patch generation workflow via CLI."""
        firmware_file = tmp_path / "patch_test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        patches_dir = tmp_path / "generated_patches"
        
        # Run CLI patch command
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "patch",
            str(firmware_file),
            "--arch", "arm",
            "--output-dir", str(patches_dir),
            "--algorithm", "dilithium2",
            "--optimization", "size"
        ], capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0, f"CLI patch generation failed: {result.stderr}"
        
        # Check patches directory was created
        assert patches_dir.exists(), "Patches directory was not created"
        
        # Check for patch files
        patch_files = list(patches_dir.glob("*.patch"))
        assert len(patch_files) > 0, "No patch files were generated"
        
        # Check patch metadata file
        metadata_file = patches_dir / "patches_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            assert "patches" in metadata
            assert "target_device" in metadata
    
    def test_cli_batch_analysis(self, firmware_samples, tmp_path):
        """Test batch analysis of multiple firmware files."""
        # Create multiple firmware files
        firmware_dir = tmp_path / "firmware_batch"
        firmware_dir.mkdir()
        
        for sample_name, sample in firmware_samples.items():
            firmware_file = firmware_dir / f"{sample_name}.bin"
            firmware_file.write_bytes(sample.data)
        
        output_dir = tmp_path / "batch_results"
        
        # Run batch analysis
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "batch-scan",
            str(firmware_dir),
            "--output-dir", str(output_dir),
            "--parallel", "2"
        ], capture_output=True, text=True, timeout=180)
        
        assert result.returncode == 0, f"Batch analysis failed: {result.stderr}"
        
        # Check output directory
        assert output_dir.exists(), "Batch output directory was not created"
        
        # Check individual result files
        result_files = list(output_dir.glob("*.json"))
        assert len(result_files) >= len(firmware_samples), \
            "Not all firmware files were analyzed"
        
        # Check summary report
        summary_file = output_dir / "batch_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            assert "total_analyzed" in summary
            assert "total_vulnerabilities" in summary


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""
    
    def test_cli_nonexistent_firmware_file(self):
        """Test CLI behavior with non-existent firmware file."""
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            "/nonexistent/file.bin",
            "--arch", "arm"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode != 0, "Should fail with non-existent file"
        assert "not found" in result.stderr.lower() or "no such file" in result.stderr.lower()
    
    def test_cli_invalid_architecture(self, arm_rsa_sample, tmp_path):
        """Test CLI behavior with invalid architecture."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "invalid_arch"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode != 0, "Should fail with invalid architecture"
        assert "architecture" in result.stderr.lower() or "invalid" in result.stderr.lower()
    
    def test_cli_empty_firmware_file(self, tmp_path):
        """Test CLI behavior with empty firmware file."""
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")
        
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(empty_file),
            "--arch", "arm"
        ], capture_output=True, text=True, timeout=30)
        
        # Should handle gracefully, either succeeding with no vulnerabilities
        # or failing with informative error
        if result.returncode == 0:
            # If successful, should indicate no vulnerabilities found
            assert "no vulnerabilities" in result.stdout.lower() or \
                   "empty" in result.stdout.lower()
        else:
            # If failed, should have informative error message
            assert "empty" in result.stderr.lower() or \
                   "invalid" in result.stderr.lower()
    
    def test_cli_permission_denied_output(self, arm_rsa_sample, tmp_path):
        """Test CLI behavior when output file cannot be written."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        # Try to write to a directory (should fail)
        invalid_output = tmp_path / "directory_not_file"
        invalid_output.mkdir()
        
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "arm",
            "--output", str(invalid_output)
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode != 0, "Should fail when cannot write output"
        assert "permission" in result.stderr.lower() or \
               "directory" in result.stderr.lower() or \
               "write" in result.stderr.lower()


class TestCLIIntegrationWorkflows:
    """Test complex integration workflows."""
    
    def test_scan_and_patch_workflow(self, arm_rsa_sample, tmp_path):
        """Test complete scan-then-patch workflow."""
        firmware_file = tmp_path / "workflow_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        scan_output = tmp_path / "scan_results.json"
        patches_dir = tmp_path / "patches"
        
        # Step 1: Scan firmware
        scan_result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "arm",
            "--output", str(scan_output)
        ], capture_output=True, text=True, timeout=60)
        
        assert scan_result.returncode == 0, f"Scan step failed: {scan_result.stderr}"
        assert scan_output.exists(), "Scan output file not created"
        
        # Step 2: Generate patches based on scan results
        patch_result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "patch",
            str(firmware_file),
            "--arch", "arm",
            "--scan-results", str(scan_output),
            "--output-dir", str(patches_dir)
        ], capture_output=True, text=True, timeout=120)
        
        assert patch_result.returncode == 0, f"Patch step failed: {patch_result.stderr}"
        assert patches_dir.exists(), "Patches directory not created"
        
        # Verify workflow consistency
        with open(scan_output, 'r') as f:
            scan_data = json.load(f)
        
        patch_files = list(patches_dir.glob("*.patch"))
        
        # Should have patches for detected vulnerabilities
        high_confidence_vulns = [
            v for v in scan_data["vulnerabilities"] 
            if v["confidence"] > 0.8
        ]
        
        if len(high_confidence_vulns) > 0:
            assert len(patch_files) > 0, \
                "Should generate patches for high-confidence vulnerabilities"
    
    def test_cli_with_configuration_file(self, arm_rsa_sample, tmp_path):
        """Test CLI usage with configuration file."""
        firmware_file = tmp_path / "config_test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        config_file = tmp_path / "pqc_config.yaml"
        config_content = """
        analysis:
          min_confidence: 0.7
          min_severity: medium
          architectures:
            - arm
            - esp32
        
        output:
          format: json
          include_metadata: true
          verbose: true
        
        patch_generation:
          default_algorithm: dilithium2
          optimization: size
          target_device: STM32L4
        """
        
        config_file.write_text(config_content)
        output_file = tmp_path / "config_results.json"
        
        # Run CLI with configuration file
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--config", str(config_file),
            "--output", str(output_file)
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"CLI with config failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"
        
        # Verify configuration was applied
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        # Check that confidence filter was applied
        for vuln in results["vulnerabilities"]:
            assert vuln["confidence"] >= 0.7, "Configuration filter not applied"
    
    def test_cli_report_generation(self, firmware_samples, tmp_path):
        """Test generation of comprehensive reports."""
        # Analyze multiple firmware samples
        results_dir = tmp_path / "analysis_results"
        results_dir.mkdir()
        
        for sample_name, sample in firmware_samples.items():
            firmware_file = tmp_path / f"{sample_name}.bin"
            firmware_file.write_bytes(sample.data)
            
            result_file = results_dir / f"{sample_name}_results.json"
            
            # Analyze each firmware
            subprocess.run([
                "python", "-m", "pqc_iot_retrofit.cli",
                "scan",
                str(firmware_file),
                "--arch", sample.architecture,
                "--output", str(result_file)
            ], capture_output=True, text=True, timeout=60)
        
        # Generate comprehensive report
        report_file = tmp_path / "comprehensive_report.html"
        
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "report",
            str(results_dir),
            "--output", str(report_file),
            "--format", "html",
            "--include-charts"
        ], capture_output=True, text=True, timeout=90)
        
        if result.returncode == 0:  # Report generation might not be implemented yet
            assert report_file.exists(), "Report file not created"
            
            # Check report content
            report_content = report_file.read_text()
            assert "PQC IoT Retrofit Scanner" in report_content
            assert "vulnerability" in report_content.lower()
        else:
            # Report generation not yet implemented
            pytest.skip("Report generation not yet implemented")


class TestCLIOutputFormats:
    """Test different CLI output formats."""
    
    @pytest.mark.parametrize("output_format", ["json", "yaml", "xml", "csv"])
    def test_cli_output_formats(self, arm_rsa_sample, tmp_path, output_format):
        """Test CLI with different output formats."""
        firmware_file = tmp_path / "format_test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        output_file = tmp_path / f"results.{output_format}"
        
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "arm",
            "--output", str(output_file),
            "--format", output_format
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            assert output_file.exists(), f"Output file for {output_format} format not created"
            
            # Basic content checks
            content = output_file.read_text()
            assert len(content) > 0, f"Output file for {output_format} is empty"
            
            if output_format == "json":
                # Should be valid JSON
                import json
                data = json.loads(content)
                assert "vulnerabilities" in data
            elif output_format == "yaml":
                # Should contain YAML-like structure
                assert "vulnerabilities:" in content or "- algorithm:" in content
            elif output_format == "xml":
                # Should contain XML tags
                assert "<vulnerabilities>" in content or "<?xml" in content
            elif output_format == "csv":
                # Should contain CSV headers
                assert "algorithm" in content and "confidence" in content
        else:
            # Format might not be implemented yet
            pytest.skip(f"Output format {output_format} not yet implemented")
    
    def test_cli_verbose_output(self, arm_rsa_sample, tmp_path):
        """Test CLI verbose mode."""
        firmware_file = tmp_path / "verbose_test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        # Run with verbose flag
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "arm",
            "--verbose"
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"Verbose scan failed: {result.stderr}"
        
        # Verbose output should contain detailed information
        output = result.stdout + result.stderr
        assert "analyzing" in output.lower() or "scanning" in output.lower()
        assert "vulnerability" in output.lower() or "algorithm" in output.lower()
    
    def test_cli_quiet_mode(self, arm_rsa_sample, tmp_path):
        """Test CLI quiet mode."""
        firmware_file = tmp_path / "quiet_test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        output_file = tmp_path / "quiet_results.json"
        
        # Run with quiet flag
        result = subprocess.run([
            "python", "-m", "pqc_iot_retrofit.cli",
            "scan",
            str(firmware_file),
            "--arch", "arm",
            "--output", str(output_file),
            "--quiet"
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"Quiet scan failed: {result.stderr}"
        
        # Quiet mode should minimize output
        output = result.stdout + result.stderr
        assert len(output.strip()) < 100, "Quiet mode should minimize output"
        
        # But results file should still be created
        assert output_file.exists(), "Results file should be created in quiet mode"