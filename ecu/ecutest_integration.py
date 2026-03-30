"""
ecu/ecutest_integration.py

Real ECU.test integration using official TraceTronic APIs:
  - REST API   http://127.0.0.1:5050/api/v2  — remote control & execution
  - Object API ApiClient / PackageApi         — create .pkg packages programmatically

References:
  https://docs.tracetronic.com/help/ecu.test/api/general_api/rest-api.html
  https://docs.tracetronic.com/help/ecu.test/api/general_api/objectApi.html
  https://docs.tracetronic.com/help/ecu.test/api/general_api/PackageApi.html
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger


# ── ECU.test REST API Client ──────────────────────────────────────────────────

class ECUTestRESTClient:
    """
    Wraps the ECU.test OpenAPI REST API (v2).
    Base URL:  http://127.0.0.1:5050/api/v2
    Swagger UI: http://127.0.0.1:5050/api/v2/ui

    Start ECU.test with:
        ECU-TEST.exe --restApiPort 5050
        # For remote access:
        ECU-TEST.exe --restApiPort 5050 --restApiEnableRemoteAccess
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5050):
        self.base_url      = f"http://{host}:{port}/api/v2"
        self.live_url      = f"{self.base_url}/live"
        self.config_url    = f"{self.base_url}/configuration"
        self.execution_url = f"{self.base_url}/execution"
        self.reports_url   = f"{self.base_url}/reports"

    def is_available(self) -> bool:
        try:
            r = requests.get(self.live_url, timeout=3)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def wait_for_operation(self, info_endpoint: str, poll_interval: float = 1.0) -> dict:
        """
        Poll endpoint until status leaves WAITING / RUNNING.
        Exact pattern from official ECU.test REST API docs.
        """
        while True:
            info = requests.get(info_endpoint, timeout=10)
            info.raise_for_status()
            status = info.json().get("status", {}).get("key", "UNKNOWN")
            logger.debug(f"  ECU.test status: {status}")
            if status not in ["WAITING", "RUNNING"]:
                logger.info(f"  Finished: {status}")
                return info.json()
            time.sleep(poll_interval)

    def load_configuration(self, tbc_path: str, tcf_path: str) -> dict:
        """PUT /configuration — load TBC + TCF and wait for completion."""
        logger.info(f"Loading TBC: {tbc_path}")
        logger.info(f"Loading TCF: {tcf_path}")
        order = {
            "action": "Start",
            "tbc": {"tbcPath": tbc_path},
            "tcf": {"tcfPath": tcf_path},
        }
        r = requests.put(self.config_url, json=order, timeout=30)
        r.raise_for_status()
        return self.wait_for_operation(self.config_url)

    def execute_package(self, package_path: str, variables: dict = None) -> dict:
        """PUT /execution — execute a .pkg and wait for result."""
        logger.info(f"Executing: {package_path}")
        order = {"testCasePath": package_path}
        if variables:
            order["variables"] = [{"name": k, "value": str(v)} for k, v in variables.items()]
        r = requests.put(self.execution_url, json=order, timeout=30)
        r.raise_for_status()
        return self.wait_for_operation(self.execution_url)

    def execute_project(self, project_path: str) -> dict:
        """PUT /execution — execute a full .prj project."""
        logger.info(f"Executing project: {project_path}")
        r = requests.put(self.execution_url, json={"testCasePath": project_path}, timeout=30)
        r.raise_for_status()
        return self.wait_for_operation(self.execution_url)

    def upload_report(self, report_id: str, testguide_url: str, auth_key: str, project_id: int) -> dict:
        """PUT /reports/{reportId}/upload — upload to test.guide."""
        logger.info(f"Uploading report {report_id} to test.guide")
        upload_url = f"{self.reports_url}/{report_id}/upload"
        r = requests.put(upload_url, json={
            "testGuideUrl": testguide_url,
            "authKey": auth_key,
            "projectId": project_id,
        }, timeout=30)
        r.raise_for_status()
        return self.wait_for_operation(upload_url)


# ── ECU.test Object API — Package Generator ───────────────────────────────────

class ECUTestPackageGenerator:
    """
    Creates ECU.test .pkg packages using the Object API PackageApi.
    Falls back to ECU.test-compatible XML when ECU.test is not installed.

    Object API requires:
        PYTHONPATH includes: <ECU.test install>/Templates/ApiClient
        import ApiClient
        api = ApiClient.ApiClient()

    Reference: https://docs.tracetronic.com/help/ecu.test/api/general_api/PackageApi.html
    """

    def __init__(self, ecutest_install_path: str = "C:/Program Files/TraceTronic/ECU-TEST"):
        self.install_path = Path(ecutest_install_path)
        self.api_client_path = self.install_path / "Templates" / "ApiClient"
        self._api_available = False
        self._api = None
        self._try_load_object_api()

    def _try_load_object_api(self):
        if self.api_client_path.exists():
            if str(self.api_client_path) not in sys.path:
                sys.path.insert(0, str(self.api_client_path))
            try:
                import ApiClient  # noqa
                self._api = ApiClient.ApiClient()
                self._api_available = True
                version = self._api.GetApplicationVersion()
                logger.info(f"ECU.test Object API loaded — version: {version}")
            except ImportError:
                logger.warning("ECU.test Object API not importable — using XML fallback")
        else:
            logger.info(f"ECU.test not found at {self.api_client_path} — using XML fallback")

    @property
    def is_available(self) -> bool:
        return self._api_available

    def create_package_via_object_api(self, tc_row: dict, output_path: str) -> bool:
        """
        Use PackageApi to build a real .pkg with:
          TsWrite  — set stimulus signals / qualifiers
          TsWait   — cycle time delay
          TsRead   — verify output with ExpectationExpression
          TsBusMonitoring — CAN bus check (ECU integration)
          TsDiagnostics — OBD/UDS steps

        PackageApi.CreatePackage(path) → Package
        Package.TestStepApi.AddTestStep(type) → TestStep
        TestStep.Signal, .Value, .ExpectationExpression, .Description
        Package.Save(path)
        """
        if not self._api_available:
            return False

        pkg_api = self._api.PackageApi
        signals    = [s.strip() for s in str(tc_row.get("Stimulus_Signals","")).split(",") if s.strip() not in ("","N/A")]
        qualifiers = [q.strip() for q in str(tc_row.get("Qualifiers","")).split(",") if q.strip() not in ("","N/A")]
        thresholds = [t.strip() for t in str(tc_row.get("Thresholds_C","")).split("/") if t.strip() not in ("","N/A")]
        tc_type    = str(tc_row.get("Integration_Type_EN", tc_row.get("Type_EN","")))

        try:
            package = pkg_api.CreatePackage(str(output_path))
            package.Description = tc_row.get("Objective_EN", "")
            ts_api = package.TestStepApi

            # Preconditions: set qualifiers OK
            for qual in qualifiers[:2]:
                s = ts_api.AddTestStep("TsWrite")
                s.Signal, s.Value = qual, "QUAL_INT_OK"
                s.Description = f"Precondition: {qual} = QUAL_INT_OK"

            # Stimulus: write input signals
            for sig in signals[:3]:
                s = ts_api.AddTestStep("TsWrite")
                s.Signal, s.Value = sig, "stimulus_value"
                s.Description = f"Stimulus: apply value to {sig}"

            # Wait one cycle
            w = ts_api.AddTestStep("TsWait")
            w.Duration = "100"
            w.Description = "Wait 100 ms for ECU processing"

            # Verification
            if "Boundary" in tc_type or "Grenzwert" in tc_type:
                for thr in thresholds[:2]:
                    r = ts_api.AddTestStep("TsRead")
                    r.Signal = signals[0] if signals else "output_signal"
                    r.ExpectationExpression = f"abs(value - {thr}) <= 0.5"
                    r.Description = f"Verify output within ±0.5°C of boundary {thr}°C"

            elif "Fault" in tc_type or "Fehlerfall" in tc_type:
                for qual in qualifiers[:1]:
                    s = ts_api.AddTestStep("TsWrite")
                    s.Signal, s.Value = qual, "QUAL_INT_INIT"
                    s.Description = f"Inject fault: {qual} = QUAL_INT_INIT"
                w2 = ts_api.AddTestStep("TsWait")
                w2.Duration = "100"
                r = ts_api.AddTestStep("TsRead")
                r.Signal = signals[0] if signals else "output_signal"
                r.ExpectationExpression = "value == BMW_LIM_MAXERRTEMP_SC"
                r.Description = "Verify error default value on fault"

            elif "CAN" in tc_type:
                b = ts_api.AddTestStep("TsBusMonitoring")
                b.Description = "Verify CAN signal transmission: ID, DLC, cycle time"

            elif "OBD" in tc_type or "Diagnostic" in tc_type:
                d = ts_api.AddTestStep("TsDiagnostics")
                d.Description = "UDS ReadDataByIdentifier (0x22) — verify PID response"

            else:
                r = ts_api.AddTestStep("TsRead")
                r.Signal = signals[0] if signals else "output_signal"
                r.ExpectationExpression = "value == expected_output"
                r.Description = tc_row.get("Expected_Result_EN", "")[:100]

            package.Save(str(output_path))
            logger.success(f"Package saved (Object API): {output_path}")
            return True

        except Exception as e:
            logger.error(f"Object API failed: {e}")
            return False

    def create_package_xml(self, tc_row: dict, output_path: str) -> str:
        """
        Generate ECU.test-compatible .pkg XML — used when ECU.test is not installed.
        XML format matches what ECU.test Object API produces internally.
        """
        tc_id      = tc_row.get("TC_ID", "TC_UNKNOWN")
        section    = tc_row.get("Section", "")
        obj_en     = tc_row.get("Objective_EN", "")
        pre_en     = tc_row.get("Preconditions_EN", "")
        exp_en     = tc_row.get("Expected_Result_EN", "")
        tc_type    = tc_row.get("Integration_Type_EN", tc_row.get("Type_EN", ""))
        priority   = tc_row.get("Priority", "High")
        signals    = [s.strip() for s in str(tc_row.get("Stimulus_Signals","")).split(",") if s.strip() not in ("","N/A")]
        thresholds = [t.strip() for t in str(tc_row.get("Thresholds_C","")).split("/") if t.strip() not in ("","N/A")]
        qualifiers = [q.strip() for q in str(tc_row.get("Qualifiers","")).split(",") if q.strip() not in ("","N/A")]

        steps = []

        # Precondition steps — set qualifiers
        for qual in qualifiers[:2]:
            steps.append(f"""
        <TestStep type="TsWrite" id="pre_{qual[-20:].replace('.','_')}">
            <Signal>{qual}</Signal><Value>QUAL_INT_OK</Value>
            <Description>Precondition: {qual} = QUAL_INT_OK</Description>
        </TestStep>""")

        # Stimulus steps
        for sig in signals[:3]:
            steps.append(f"""
        <TestStep type="TsWrite" id="stim_{sig[-20:].replace('.','_')}">
            <Signal>{sig}</Signal><Value>stimulus_value</Value>
            <Description>Apply stimulus to {sig}</Description>
        </TestStep>""")

        # Wait cycle
        steps.append("""
        <TestStep type="TsWait" id="wait_cycle">
            <Duration>100</Duration>
            <Description>Wait 100 ms for ECU signal propagation</Description>
        </TestStep>""")

        # Verification
        if "Boundary" in tc_type or "Grenzwert" in tc_type:
            for i, thr in enumerate(thresholds[:2]):
                steps.append(f"""
        <TestStep type="TsRead" id="verify_boundary_{i}">
            <Signal>{signals[0] if signals else 'output_signal'}</Signal>
            <ExpectationExpression>abs(value - {thr}) &lt;= 0.5</ExpectationExpression>
            <Description>Verify output at boundary {thr}°C (±0.5°C tolerance)</Description>
        </TestStep>""")

        elif "Fault" in tc_type or "Fehlerfall" in tc_type:
            for qual in qualifiers[:1]:
                steps.append(f"""
        <TestStep type="TsWrite" id="inject_fault">
            <Signal>{qual}</Signal><Value>QUAL_INT_INIT</Value>
            <Description>Inject fault: {qual} = QUAL_INT_INIT</Description>
        </TestStep>""")
            steps.append("""
        <TestStep type="TsWait" id="wait_fault"><Duration>100</Duration></TestStep>""")
            steps.append(f"""
        <TestStep type="TsRead" id="verify_fault">
            <Signal>{signals[0] if signals else 'output_signal'}</Signal>
            <ExpectationExpression>value == BMW_LIM_MAXERRTEMP_SC</ExpectationExpression>
            <Description>Verify error default value is applied on fault</Description>
        </TestStep>""")

        elif "Out-of-Range" in tc_type or "Bereichsüberschreitung" in tc_type:
            steps.append(f"""
        <TestStep type="TsWrite" id="oor_stimulus">
            <Signal>{signals[0] if signals else 'input_signal'}</Signal>
            <Value>999</Value>
            <Description>Apply out-of-range stimulus (999°C)</Description>
        </TestStep>""")
            steps.append("""
        <TestStep type="TsWait" id="wait_oor"><Duration>100</Duration></TestStep>""")
            steps.append(f"""
        <TestStep type="TsRead" id="verify_oor">
            <Signal>{signals[0] if signals else 'output_signal'}</Signal>
            <ExpectationExpression>value == BMW_LIM_MAXERRTEMP_SC</ExpectationExpression>
            <Description>Verify output blocked at error/default value</Description>
        </TestStep>""")

        elif "CAN" in tc_type:
            steps.append(f"""
        <TestStep type="TsBusMonitoring" id="can_monitor">
            <BusType>CAN</BusType>
            <Signal>{signals[0] if signals else 'CAN_signal'}</Signal>
            <ExpectedCycleTime>100</ExpectedCycleTime>
            <TolerancePct>10</TolerancePct>
            <Description>Verify CAN signal: ID, DLC, value, cycle time ±10%</Description>
        </TestStep>""")

        elif "OBD" in tc_type:
            steps.append("""
        <TestStep type="TsDiagnostics" id="obd_request">
            <Service>0x22</Service>
            <Description>UDS ReadDataByIdentifier — verify PID response within 50ms</Description>
            <ExpectedResponseTime>50</ExpectedResponseTime>
        </TestStep>""")

        elif "DTC" in tc_type:
            steps.append("""
        <TestStep type="TsDiagnostics" id="dtc_read">
            <Service>0x19</Service><SubFunction>0x02</SubFunction>
            <Description>Read DTC — UDS 0x19 0x02; expect status byte 0x09</Description>
            <ExpectedStatusByte>0x09</ExpectedStatusByte>
        </TestStep>""")

        else:
            steps.append(f"""
        <TestStep type="TsRead" id="verify_nominal">
            <Signal>{signals[0] if signals else 'output_signal'}</Signal>
            <ExpectationExpression>value == expected_output</ExpectationExpression>
            <Description>{(exp_en or '')[:120]}</Description>
        </TestStep>""")

        var_lines = "\n        ".join(
            [f'<Variable name="{s}" type="Signal" direction="Input"/>' for s in signals[:4]] +
            [f'<Variable name="{q}" type="Qualifier" direction="Input"/>' for q in qualifiers[:2]] +
            ['<Variable name="expected_output" type="Value" direction="Input" default="0"/>',
             '<Variable name="error_default_value" type="Value" direction="Input" default="BMW_LIM_MAXERRTEMP_SC"/>']
        )

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- ECU.test Package — Generated by BMS GenAI Assistant -->
<!-- Object API ref: https://docs.tracetronic.com/help/ecu.test/api/general_api/PackageApi.html -->
<Package version="2024.1">
    <Attributes>
        <Id>{tc_id}</Id>
        <Description>{obj_en}</Description>
        <RequirementSection>{section}</RequirementSection>
        <TestType>{tc_type}</TestType>
        <Priority>{priority}</Priority>
    </Attributes>
    <Variables>
        {var_lines}
    </Variables>
    <Preconditions><Description>{pre_en}</Description></Preconditions>
    <TestSteps>{"".join(steps)}
    </TestSteps>
    <ExpectedResult><Description>{exp_en}</Description></ExpectedResult>
</Package>
"""
        Path(output_path).write_text(xml, encoding="utf-8")
        return output_path


# ── Full Suite Generator ───────────────────────────────────────────────────────

class BMSECUTestSuiteGenerator:
    """
    Generates a complete ECU.test test suite:
      - .pkg per ECU integration test case (Object API or XML)
      - .prj project containing all packages
      - .tbc stub (test bench config)
      - .tcf stub (test configuration)
      - run_ecutest.py REST runner script
    """

    def __init__(
        self,
        ecutest_host: str = "127.0.0.1",
        ecutest_port: int = 5050,
        ecutest_install_path: str = "C:/Program Files/TraceTronic/ECU-TEST",
    ):
        self.rest  = ECUTestRESTClient(ecutest_host, ecutest_port)
        self.gen   = ECUTestPackageGenerator(ecutest_install_path)

    def generate_suite(self, df_ecu: pd.DataFrame, output_dir: str = "./outputs/ecutest") -> dict:
        out = Path(output_dir)
        pkg_dir = out / "Packages"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        packages = []
        for _, row in df_ecu.iterrows():
            tc_id = str(row.get("TC_ID", "TC")).replace("/", "_")
            pkg_path = pkg_dir / f"{tc_id}.pkg"
            if self.gen.is_available:
                ok = self.gen.create_package_via_object_api(row.to_dict(), str(pkg_path))
                if not ok:
                    self.gen.create_package_xml(row.to_dict(), str(pkg_path))
            else:
                self.gen.create_package_xml(row.to_dict(), str(pkg_path))
            packages.append(str(pkg_path))

        prj  = self._generate_project(packages, out)
        tbc  = self._generate_tbc(out)
        tcf  = self._generate_tcf(out)
        runner = self._generate_runner(prj, tbc, tcf, out)

        logger.success(f"ECU.test suite generated: {out}")
        logger.info(f"  Packages: {len(packages)}  |  Runner: {runner}")
        return {"output_dir": str(out), "packages": packages,
                "project": prj, "tbc": tbc, "tcf": tcf, "runner": runner}

    def _generate_project(self, package_paths: list, out: Path) -> str:
        prj = out / "BMS_ECU_Integration_Tests.prj"
        entries = "\n".join(
            f'    <TestCase path="Packages/{Path(p).name}" active="true"/>'
            for p in package_paths
        )
        prj.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- ECU.test Project — BMS Integration Tests -->
<!-- Execute via REST API: PUT /execution {{"testCasePath": "{prj.name}"}} -->
<Project version="2024.1">
    <n>BMS_ECU_Integration_Tests</n>
    <Description>Auto-generated BMS ECU Integration Test Suite</Description>
    <TestCases>
{entries}
    </TestCases>
    <ExecutionConfig>
        <StopOnError>false</StopOnError>
        <RecordingMode>Always</RecordingMode>
    </ExecutionConfig>
</Project>
""", encoding="utf-8")
        return str(prj)

    def _generate_tbc(self, out: Path) -> str:
        tbc = out / "BMS_HIL_Bench.tbc"
        tbc.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!-- ECU.test Test Bench Configuration (TBC) -->
<!-- Configure your HIL bench, CAN interfaces, and diagnostic connections -->
<TestBenchConfiguration version="2024.1">
    <n>BMS_HIL_Bench</n>
    <Description>BMS HIL Test Bench — configure for your HIL platform</Description>
    <Platform>
        <!-- <ModelAccess name="Plant model" file="BMS_Plant_Model.mdl"/> -->
    </Platform>
    <Buses>
        <!-- <CANBus name="powertrain_CAN" dbcFile="BMS_CAN.dbc" channel="1"/> -->
    </Buses>
    <Diagnostics>
        <!-- <DiagInterface type="PEAK" channel="1" protocol="UDS"/> -->
    </Diagnostics>
</TestBenchConfiguration>
""", encoding="utf-8")
        return str(tbc)

    def _generate_tcf(self, out: Path) -> str:
        tcf = out / "BMS_Test_Config.tcf"
        tcf.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!-- ECU.test Test Configuration (TCF) -->
<!-- Configure SUT (ECU), signal mappings, and recording settings -->
<TestConfiguration version="2024.1">
    <n>BMS_Test_Config</n>
    <Description>BMS ECU Test Configuration</Description>
    <SUT>
        <!-- <ECU name="BMS_ECU" interface="XCP_ETH" address="192.168.1.100"/> -->
    </SUT>
    <GlobalMappings>
        <!-- <Mapping logical="BMW_t_HvsCellMinObd" physical="HIL.BMS.CellMinTemp"/> -->
    </GlobalMappings>
    <Recording>
        <Mode>Always</Mode>
    </Recording>
</TestConfiguration>
""", encoding="utf-8")
        return str(tcf)

    def _generate_runner(self, prj: str, tbc: str, tcf: str, out: Path) -> str:
        runner = out / "run_ecutest.py"
        runner.write_text(f'''"""
ECU.test REST API Runner — BMS Integration Tests
Auto-generated by BMS GenAI Assistant

Reference: https://docs.tracetronic.com/help/ecu.test/api/general_api/rest-api.html

Prerequisites:
  1. Start ECU.test: ECU-TEST.exe --restApiPort 5050
  2. python run_ecutest.py
"""

from time import sleep
import requests, sys

BASE_URL    = "http://127.0.0.1:5050/api/v2"
CONFIG_URL  = f"{{BASE_URL}}/configuration"
EXEC_URL    = f"{{BASE_URL}}/execution"
REPORTS_URL = f"{{BASE_URL}}/reports"
TBC_PATH    = r"{tbc}"
TCF_PATH    = r"{tcf}"
PRJ_PATH    = r"{prj}"
TESTGUIDE_URL = ""   # optional: "https://my-testguide:1234"
TESTGUIDE_KEY = ""
TESTGUIDE_PID = 1


def wait(endpoint):
    while True:
        r = requests.get(endpoint, timeout=10)
        s = r.json().get("status", {{}}).get("key", "UNKNOWN")
        print(f"  status: {{s}}")
        if s not in ["WAITING", "RUNNING"]:
            return r.json()
        sleep(1)


def main():
    try:
        requests.get(f"{{BASE_URL}}/live", timeout=3)
        print("ECU.test REST API reachable ✅")
    except Exception:
        print("Cannot connect — start ECU.test with: ECU-TEST.exe --restApiPort 5050")
        sys.exit(1)

    print("\\nLoading TBC + TCF...")
    r = requests.put(CONFIG_URL, json={{"action": "Start",
        "tbc": {{"tbcPath": TBC_PATH}}, "tcf": {{"tcfPath": TCF_PATH}}}}, timeout=30)
    r.raise_for_status()
    wait(CONFIG_URL)

    print(f"\\nExecuting: {{PRJ_PATH}}")
    r = requests.put(EXEC_URL, json={{"testCasePath": PRJ_PATH}}, timeout=30)
    r.raise_for_status()
    result = wait(EXEC_URL)
    verdict   = result.get("result", {{}}).get("reportStatus", "UNKNOWN")
    report_id = result.get("result", {{}}).get("testReportId", "")
    print(f"\\nVerdict: {{verdict}}  |  Report ID: {{report_id}}")

    if TESTGUIDE_URL and TESTGUIDE_KEY and report_id:
        print("Uploading to test.guide...")
        up = requests.put(f"{{REPORTS_URL}}/{{report_id}}/upload",
            json={{"testGuideUrl": TESTGUIDE_URL, "authKey": TESTGUIDE_KEY,
                   "projectId": TESTGUIDE_PID}}, timeout=30)
        up.raise_for_status()
        wait(f"{{REPORTS_URL}}/{{report_id}}/upload")

    print(f"\\nDone — {{verdict}}")


if __name__ == "__main__":
    main()
''', encoding="utf-8")
        return str(runner)

    def run_suite(self, tbc_path: str, tcf_path: str, project_path: str) -> dict:
        """Execute via REST API if ECU.test is running."""
        if not self.rest.is_available():
            logger.warning("ECU.test not running — skipping execution")
            return {"status": "skipped", "reason": "ECU.test REST API not reachable"}
        cfg = self.rest.load_configuration(tbc_path, tcf_path)
        exc = self.rest.execute_project(project_path)
        return {
            "status":    "complete",
            "verdict":   exc.get("result", {}).get("reportStatus", "UNKNOWN"),
            "report_id": exc.get("result", {}).get("testReportId", ""),
        }
